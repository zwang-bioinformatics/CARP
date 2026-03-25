###################################

import os
os.environ['LD_LIBRARY_PATH'] = "/home/asiciliano/anaconda3/envs/analysis/lib"
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numexpr as ne 
import pandas as pd
import math
import numpy as np
from scipy.stats import gaussian_kde, zscore
from tqdm import tqdm
import json
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from termcolor import colored
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import scienceplots
import argparse

import sys
sys.path.append(f"../visualization/")
from viz_util import *

from multiprocessing import Pool
from more_itertools import chunked

###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--dataset", action="store", type=str, help='''
        The dataset_csv file to use for plotting.  
        ''',
        default = "/home/asiciliano/CARP/data/datasets/DOCKING.csv"
    )
    parser.add_argument("-p","--procs", action="store", type=int, help='''
        Number of processes to use for gathering predictions
        ''',
        default = 1
    )
    parser.add_argument("-f","--dist_factor", action="store", type=int, help='''
        factor for updating multiprocessing progress bar, lower rates give faster updates
        ''',
        default = 0.01
    )

    args = parser.parse_args()

    ###################
    dataset_tag = args.dataset.split("/")[-1][:-4]
    irmsd_key = "irmsd_protein_fit" if "DOCKING" in dataset_tag else "irmsd_interface_fit"

    for dset in [args.dataset, "/".join(args.dataset.split("/")[:-1]) + f"/perturbed_{dataset_tag}.csv"]:

        dset_tag = dset.split("/")[-1][:-4]
        dataset = pd.read_csv(dset)

        with tqdm(total=len(dataset),desc="Reading Dataset") as pbar:

            to_gather = []
            
            for target_src, model_df in dataset.groupby("target_src"):
                if not all([
                    (
                        # See: https://colab.research.google.com/drive/1XQ4-WRKAOXa81XZbcEosrG7dS09gc8MI?usp=sharing
                        os.path.exists(target_src + "/nsp/01/01.csv") or os.path.exists(target_src + "/nsp/results.csv") # server output?
                    ), 
                    os.path.exists(target_src + "bp.mat")
                ]): 
                    pbar.update(len(model_df))
                    continue
                
                for _, row in model_df.iterrows():
                    if not os.path.exists(f"{row['model_src']}/model.pdb"): pbar.update(1); continue
                    if not os.path.exists(f"{row['model_src']}/predicted_quality/carp.csv"): pbar.update(1); continue

                    to_gather += [(target_src.split("/")[-2], target_src, row["model_src"])]
                    pbar.update(1)

                # break

        ###################

        results = {"global": [], "interface": []}

        if args.procs == 1: results, _ = gather(tqdm(to_gather, desc="Gathering Predictions"), include_other=False)
        else:
            chunk_size = math.ceil( args.dist_factor * (len(to_gather)/args.procs) )

            with tqdm(total=len(to_gather), desc="Gathering Predictions") as pbar:

                pool = Pool(processes=args.procs)

                jobs = [
                    pool.apply_async(
                        gather, 
                        args=(chunk,False), 
                        callback=lambda r: pbar.update(r[1])
                    ) 
                    for chunk in chunked(to_gather, chunk_size)
                ]

                pool.close()
                pool.join()

                for j in jobs: 
                    r, _ = j.get()
                    for key in r: results[key] += r[key]

        for key in results:
            if not len(results[key]): continue
            results[key] = pd.DataFrame(results[key])

        ###################

        results["global"]["irmsd_quantile"] = -results["global"][irmsd_key]
        # Q(irmsd) = (count(value_target \geq irmsd)) / N_target
        results["global"]["irmsd_quantile"] = 100*(results["global"].groupby("target")["irmsd_quantile"].rank(method='max', pct = True)) 
        results["global"]["success"] = results["global"]["irmsd_quantile"] >= 99
        results["global"]["num_success"] = results["global"].groupby("target")["success"].transform('sum')

        group_tag = "CARP"
        methods = [
            'CARP:Merged', 
            'CARP:Iface', 
            'CARP:Fold', 
            'CARP:RP',
        ]

        results["global"] = results["global"].melt(
            id_vars=["target", "model", "num_success", "success", "irmsd_quantile"], 
            var_name="method", 
            value_vars=methods, 
            value_name="score"
        )

        ###################

        performances = []
        ks = [5, 10, 15, 20, 25, 50, 100]

        for k in ks:
            top_k_df = results["global"].groupby(["target", "method"]).apply(lambda x: x.nlargest(k, "score")).reset_index(drop=True)
            top_k_df = top_k_df.groupby(["target", "method"]).agg(**{
                "avg_quantile": ("irmsd_quantile", "mean"),
                "max_quantile": ("irmsd_quantile", "max"),
                "success": ("success", "max"),
                "recall": ("success", "sum"),
                "num_success": ("num_success", "first")
            }).reset_index()
            top_k_df["recall"] = 100*(top_k_df["recall"] / top_k_df["num_success"])
            top_k_df["success"] = 100*top_k_df["success"]
            del top_k_df["num_success"]
            top_k_df["K"] = k
            top_k_df = top_k_df.groupby("method").mean(numeric_only=True).reset_index()
            performances += [top_k_df]

        ###################

        performances = pd.concat(performances)

        for key, label in zip(
            ["success", "recall", "avg_quantile", "max_quantile"],
            [r"Success (%)", r"Recall (%)", r"Avg Quantile (%)", r"Best Quantile (%)"],
        ):
            performances[f"rel_{key}"] = performances[key].round(2)
            performances[f"rel_{key}"] = performances.groupby("K")[f"rel_{key}"].transform(mm_scale)

            pivot_df = performances.pivot(index='method', columns='K', values=[f"rel_{key}", key])
            pivot_df.index = [format_title(m) for m in pivot_df.index]
            # print(pivot_df.index)
            # print(methods)
            pivot_df = pivot_df.reindex(map(format_title, methods))

            with plt.style.context('nature'):
                fig, ax = plt.subplots(figsize=(14, 6.75))
                hax = sns.heatmap(
                    pivot_df[f"rel_{key}"], 
                    annot=pivot_df[key], 
                    linewidth=1, cmap= "YlGnBu",
                    fmt='.3f', ax = ax,
                    cbar_kws={'format': '%.1f', 'shrink': 0.9},
                    annot_kws={"size": 18}
                )
                ax.tick_params(axis='y', labelsize=23.5, length=0, pad=8, labelrotation=0)
                ax.tick_params(axis='x', labelsize=20)

                plt.title(label, fontsize=30, pad = 25)
                plt.xlabel(r"$K$", fontsize=23.5, labelpad = 16)
                plt.ylabel("")
                
                cbar = ax.collections[0].colorbar
                cbar.ax.yaxis.set_ticks_position('left')
                cbar.set_label("Relative Performance", fontsize=21.5, labelpad = 30, rotation=-90)
                cbar.ax.tick_params(axis='y', labelsize=15)

                mval = [pivot_df[key][k].round(2).max() for k in pivot_df[key].columns]
                mnval = [pivot_df[key][k].round(2).min() for k in pivot_df[key].columns]

                os.makedirs(f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/", exist_ok=True)

                plt.tight_layout()
                plt.savefig(
                    f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/heat_{key}.png", 
                    dpi=300,
                    transparent=False,
                )
                plt.clf()
                plt.close()

        ###################

        rows, columns = 2, 2
        fig = plt.figure(figsize=(25,12.5))

        for i,image_info in enumerate([
            ("",f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/heat_success.png"),
            ("",f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/heat_recall.png"),
            ("",f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/heat_max_quantile.png"),
            ("",f"/home/asiciliano/CARP/data/figures/{dset_tag}/{group_tag}/heat_avg_quantile.png"),
        ]): 
            label, file = image_info
            ax = fig.add_subplot(rows, columns, i + 1) 
            ax.autoscale(tight=True)        
            plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)) 
            plt.axis('off') 
            # plt.title("("+string.ascii_uppercase[i]+") " + label, fontsize=20, pad=0, loc='left', y=0.97) 

        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.0, hspace=0.005)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        plt.savefig(f"/home/asiciliano/CARP/data/figures/{dset_tag}/{dset_tag}_heat_{group_tag}.png", dpi=300, bbox_inches = 'tight', transparent = False)
        plt.clf()
        plt.close()

        print(f"See @: /home/asiciliano/CARP/data/figures/{dset_tag}/{dset_tag}_heat_{group_tag}.png")

        ###################


###################################
