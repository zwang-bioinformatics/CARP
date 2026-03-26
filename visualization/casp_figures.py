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

from viz_util import *

import sys
sys.path.append(f"../")

from multiprocessing import Pool
from more_itertools import chunked

###################################

def casp_global(df):
    # assumes the df contains all models for a single target...

    mu = df[["tm", "oligo_gdtts", "bb_lddt", "ics", "ips", "ilddt"]].mean().to_dict()
    std = df[["tm", "oligo_gdtts", "bb_lddt", "ics", "ips", "ilddt"]].std().to_dict()

    for score in ["tm", "oligo_gdtts", "bb_lddt", "ics", "ips", "ilddt"]:
        msk = ((df[score] - mu[score]) / std[score]) > (-2)
        mu[score] = df[score][msk].mean()
        std[score] = df[score][msk].std()

        df[f"Z:{score}"] = (df[score] - mu[score]) / std[score]

    df["Z-CASP16"] = ( # NA-Prot, see https://doi.org/10.1002/prot.70072
        0.3*(0.3*df["Z:tm"] + 0.3*df["Z:oligo_gdtts"] + 0.4*df["Z:bb_lddt"]) +
        0.7*((df["Z:ics"] + df["Z:ips"] + df["Z:ilddt"]) / 3)
    )

    df["density"] = gaussian_kde(df["Z-CASP16"])(df["Z-CASP16"])
    best_rmsd = df["rmsd"].min()
    df["RMSD Ranking Loss"] = df["rmsd"] - df["rmsd"].min()

    return df

def filtering(df):

    passed = True

    for group, gdf in df.groupby("method"):
        if gdf["score"].isna().mean() > 0.4: passed = False

    df["passed_g"] = passed

    return df

def casp_perif(df):

    best_idx = df['score'].idxmax()

    return pd.Series({
        'Pearson ICS': df['score'].corr(df['ics']),
        'Spearman ICS': df['score'].corr(df['ics'], method='spearman'),
        'Ranking Loss ICS': df.loc[best_idx,"Ranking Loss ICS"],
        'Pearson IPS': df['score'].corr(df['ips']),
        'Spearman IPS': df['score'].corr(df['ips'], method='spearman'),
        'Ranking Loss IPS': df.loc[best_idx,"Ranking Loss IPS"],
    })

###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p","--procs", action="store", type=int, help='''
        Number of processes to use for gathering predictions
        ''',
        default = 1
    )
    parser.add_argument("-f","--dist_factor", action="store", type=int, help='''
        factor for updating multiprocessing progress bar, lower rates give faster updates
        ''',
        default = 0.1
    )

    args = parser.parse_args()

    ###################

    methods = [
        'CARP:Merged', 
        'CARP:Iface', 
        'CARP:Fold', 
        'CARP:RP',

        "DRPScore",
        "ITScore-PR",
        "3dRPC",
        "FTDMP",
        "QUASI-RNP",
        "DARS-RNP",
    ]

    methods_if = [
        'CARP:PerIface',
        'CARP:PerIface-G',

        "DRPScore",
        "ITScore-PR",
        "3dRPC",
        "FTDMP",
        "QUASI-RNP",
        "DARS-RNP",
    ]

    dataset_tag = "CASP16"
    dataset = pd.read_csv("/home/asiciliano/CARP/data/datasets/CASP16.csv")

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
                to_gather += [(target_src.split("/")[-2], target_src, row["model_src"])]
                pbar.update(1)

    ###################

    results = {"global": [], "interface": []}

    if args.procs == 1: results, _ = gather(tqdm(to_gather, desc="Gathering Predictions"))
    else:
        chunk_size = math.ceil( args.dist_factor * (len(to_gather)/args.procs) )

        with tqdm(total=len(to_gather), desc="Gathering Predictions") as pbar:

            pool = Pool(processes=args.procs)

            jobs = [
                pool.apply_async(
                    gather, 
                    args=(chunk,), 
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

    results["global"] = results["global"].groupby("target", group_keys=False).apply(casp_global)
    df_3 = results["global"].melt(
        id_vars=["target", "model", "Z-CASP16", "RMSD Ranking Loss"], 
        var_name="method", 
        value_vars=methods, 
        value_name="score"
    )
    df_3 = df_3.groupby(["target", "method"]).apply(lambda x: x.nlargest(3, "score")).reset_index(drop=True)
    df_3["K"] = df_3.groupby(["target", "method"]).cumcount() + 1
    df_3 = df_3.sort_values(['method', 'K'])

    all_sum = df_3.groupby("method")["Z-CASP16"].apply(lambda col: col[col > 0].sum()).reset_index()
    all_sum = all_sum.sort_values(by='Z-CASP16', ascending=False).reset_index()
    best = all_sum.loc[all_sum['Z-CASP16'].idxmax()]["method"]
    order = all_sum["method"].apply(lambda m: format_title(m, bold = m == best)).tolist()

    ###################

    plot_df = df_3.groupby(['method', 'K'])['Z-CASP16'].apply(
        lambda x: x[x > 0].sum()
    ).reset_index().sort_values(['method', 'K'])
    plot_df['Z_plot'] = plot_df.groupby('method')['Z-CASP16'].cumsum()

    with plt.style.context(['nature', 'std-colors']): 

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        for k in range(3, 0, -1):
            df = plot_df[plot_df["K"] == k].copy()
            df["method"] = df["method"].apply(lambda m: format_title(m, bold = m == best))

            sns.barplot(
                x='Z_plot', 
                y='method', 
                data=df,
                dodge = False,
                zorder = 4 - k,
                order = order,
                label = f"Rank {k}",
                ax = ax
            )

        ax.autoscale()
        ax.set_xlabel("Sum Z-CASP16 > 0", fontsize = 21, labelpad=18)
        ax.set_ylabel(None)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=21)
        ax.legend(
            title='Model', 
            loc = 'lower right',
            frameon = False,
            title_fontsize=18,
            fontsize = 16,
        )

        ##################

        os.makedirs(f"../data/figures/CASP16/", exist_ok=True)

        plt.tight_layout()
        plt.savefig(
            "../data/figures/CASP16/ZRANK_FULL.png",
            dpi=300, bbox_inches = 'tight', transparent = False
        )
        plt.clf()
        plt.close()
        
        print("See @: ../data/figures/CASP16/ZRANK_FULL.png")

    ###################

    results["interface"]["Ranking Loss ICS"] = results["interface"].groupby(["target", "interface"])["ics"].transform(lambda x: x.max() - x)
    results["interface"]["Ranking Loss IPS"] = results["interface"].groupby(["target", "interface"])["ips"].transform(lambda x: x.max() - x)        

    results["interface"] = results["interface"].melt(
        id_vars=["target", "model", "reference_interface", "ips", "ics", "Ranking Loss ICS", "Ranking Loss IPS"], 
        var_name="method", 
        value_vars=methods_if, 
        value_name="score"
    )

    def kill(x):
        x["passed_t"] = (not all(x["ics"].isna()) and not all(x["ips"].isna())) and len(x["ics"].unique()) > 1 and len(x["ips"].unique()) > 1
        return x

    results["interface"] = results["interface"].groupby(["target","reference_interface"], group_keys=False).apply(kill)
    results["interface"] = results["interface"][results["interface"]["passed_t"]]
    results["interface"] = results["interface"].groupby(["target","reference_interface"], group_keys=False).apply(filtering)
    results["interface"] = results["interface"][results["interface"]["passed_g"]]

    summary = results["interface"].groupby(["target", "reference_interface", "method"]).apply(
        casp_perif, include_groups=False
    ).reset_index()
    
    for key in ["ICS", "IPS"]:

        results_agg = summary.copy()
        results_agg["spearman"] = results_agg[f"Spearman {key}"].clip(0,None)
        results_agg["pearson"] = results_agg[f"Pearson {key}"].clip(0,None)
        results_agg["ranking_loss"] = results_agg[f"Ranking Loss {key}"]

        results_agg = results_agg.groupby(["method", "target"])[["spearman", "pearson", "ranking_loss"]].mean().reset_index()
        results_agg = results_agg.groupby("method")[["spearman", "pearson", "ranking_loss"]].mean().reset_index()

        results_agg["method"] = results_agg["method"].apply(format_title)
        results_agg = results_agg.set_index("method")
        results_agg = results_agg.reindex([format_title(m) for m in methods_if])

        print(key.upper(),"->\n")
        print(results_agg.round(3).to_latex(float_format="%.3f"))
        print("\n")

        results_agg = summary.copy()
        results_agg["spearman"] = results_agg[f"Spearman {key}"]
        results_agg["pearson"] = results_agg[f"Pearson {key}"]
        results_agg["ranking_loss"] = results_agg[f"Ranking Loss {key}"]
        results_agg["method"] = results_agg["method"].apply(format_title)

        for metric in ["spearman", "pearson", "ranking_loss"]:


            heat_df = results_agg.copy()
            heat_df["target"] = heat_df["target"] + ":" + heat_df["reference_interface"] 
            heat_df["z-score"] = heat_df.groupby("target")[metric].transform(mm_scale)

            heat_df["z-score"] *= (-1)**(metric == "ranking_loss")

            heat_df = heat_df.pivot(index='method', columns='target', values=[metric, "z-score"])
            formatted_methods = [format_title(m) for m in methods_if]
            heat_df = heat_df.reindex(formatted_methods)
            fig, ax = plt.subplots(figsize=(30, 6.75))

            hax = sns.heatmap(
                heat_df["z-score"], 
                annot=heat_df[metric], #.reindex_like(heat_df["z-score"]), 
                linewidth=1,
                cmap= "YlGnBu", #"cividis"
                fmt='.2f',
                ax = ax,
                cbar_kws={'format': '%.1f', 'shrink': 0.9},# 'location': 'right', 
                # norm=LogNorm(), 
                annot_kws={"size": 18}
            )
            ax.tick_params(axis='y', labelsize=23.5, length = 0, pad=8)
            ax.tick_params(axis='x', labelsize=20)

            title = metric.replace("_", " ").title()

            plt.title(title, fontsize=30, pad = 25)
            plt.xlabel(r"CASP16 Target", fontsize=28, labelpad = 25)
            plt.ylabel("")

            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.set_label("Relative Performance", fontsize=21.5, labelpad = 30, rotation=-90)
            cbar.ax.tick_params(axis='y', labelsize=15)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            ##################

            mval = [heat_df[metric][k].round(2).max() for k in heat_df[metric].columns]
            mnval = [heat_df[metric][k].round(2).min() for k in heat_df[metric].columns]

            for text in ax.texts:
                val = round(float(text.get_text()),2)

                if metric == "ranking_loss": 
                    if val == mnval[int(text._x - 0.5)] and val != mval[int(text._x - 0.5)]: 
                        text.set_text(r"$\bf{" + text.get_text() + r"}$")
                        text.set_fontsize(19.5)    
                else: 
                    if val != mnval[int(text._x - 0.5)] and val == mval[int(text._x - 0.5)]: 
                        text.set_text(r"$\bf{" + text.get_text() + r"}$")
                        text.set_fontsize(19.5)   

            ##################

            plt.tight_layout()
            plt.savefig(
                f"/home/asiciliano/CARP/data/figures/CASP16/sub/heat_casp_iface_{metric}_{key}.png",
                dpi=300, bbox_inches = 'tight', transparent = False
            )
            plt.clf()
            plt.close()

        ###################################

        rows, columns = 3, 1
        fig = plt.figure(figsize=(30, 18))

        for i,image_info in enumerate([
            ("",f"/home/asiciliano/CARP/data/figures/CASP16/sub/heat_casp_iface_pearson_{key}.png"),
            ("",f"/home/asiciliano/CARP/data/figures/CASP16/sub/heat_casp_iface_spearman_{key}.png"),
            ("",f"/home/asiciliano/CARP/data/figures/CASP16/sub/heat_casp_iface_ranking_loss_{key}.png"),

        ]): 
            label, file = image_info
            ax = fig.add_subplot(rows, columns, i + 1) 
            ax.autoscale(tight=True)        
            plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)) 
            plt.axis('off') 
            plt.title(
                "("+string.ascii_uppercase[i]+") " + label, 
                fontsize=48, pad=0, 
                loc='left', x = -0.05
            ) 

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0, hspace=0.125)
        plt.savefig(
            f"/home/asiciliano/CARP/data/figures/CASP16/heat_casp_iface_full_{key}.png", 
            dpi=300, 
            bbox_inches = 'tight', 
            transparent = False
        )
        plt.clf()
        plt.close()

        print("See:", f"/home/asiciliano/CARP/data/figures/CASP16/heat_casp_iface_full_{key}.png")

###################################
