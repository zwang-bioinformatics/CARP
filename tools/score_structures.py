###################################

# docker run -d --name ost_worker --network host --entrypoint "" -v /home/asiciliano/CARP/tools/:/root/tools -v /home/asiciliano/CARP/data/targets/:/root/targets ost_2.11.1 tail -f /dev/null

###################################

import os
import subprocess

import math
from multiprocessing import Pool
from more_itertools import chunked

import argparse
import pandas as pd
from tqdm import tqdm

###################################

def check_ost(model_src):
    if os.path.exists(model_src + ".json"): 
        try: 
            ost = json.load(open(output_user + ".json",'r'))
            test = {
                key: ost[key]
                for key in [
                    "ips", "ics","oligo_gdtts", 
                    "oligo_gdtha","lddt",
                    "bb_lddt", "ilddt", 
                    "tm","qs_best", "rmsd",  
                    "dockq_wave", "dockq_ave",
                    "mapping_coverage_interface", 
                    "irmsd_global_fit", "irmsd_interface_fit",
                    "irmsd_protein_fit","rmsd_prot",
                    "irmsd_rna_fit", "rmsd_rna"
                ]
            } 
            return True
        except: print("Investigate:", output_user + ".json")
    return False

###################################

def run_ost(tasks):
    out = {}
    
    for task in tasks:
        native_pdb, model_src = task

        if not check_ost(model_src):
            irmsd_cmd = [
                "docker", "exec", "ost_worker", "ost",
                "/root/tools/score_ost.py", # model, reference, out_json
                model_src.replace("//", "/").replace("/home/asiciliano/CARP/data/targets/", "/root/targets/") + "/model.pdb", 
                native_pdb.replace("//", "/").replace("/home/asiciliano/CARP/data/targets/", "/root/targets/"),
                model_src.replace("//", "/").replace("/home/asiciliano/CARP/data/targets/", "/root/targets/") + "/scores.json"
            ]
            # print(" ".join(irmsd_cmd),"->\n")
            output = subprocess.getoutput(" ".join(irmsd_cmd))
            # print(output)
            with open(f"{model_src}/scores.log", 'w') as f: f.write(output)

        check_ost(model_src)

    return len(tasks)

###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--dataset_csv", action="store", type=str, help='''
        The dataset_csv file to use for custom openstructure scoring. If the file '{target_src}/native.pdb' does not exist for 
        a {target_src} in the dataset, it will be skipped. The script will only score '{model_src}/model.pdb' {model_src} in the 
        dataset, if it exists, and subsequently the file '{model_src}/scores.json' will be created. 
        ''',
        default = "/home/asiciliano/CARP/data/datasets/CASP16.csv"
    )

    parser.add_argument("-p","--procs", action="store", type=int, default=1, help="number of processes to use")

    args = parser.parse_args()

    ###################################

    dataset = pd.read_csv(args.dataset_csv)

    for target_src, model_df in dataset.groupby("target_src"):

        print("#"*15)

        to_run = []

        if not os.path.exists(f"{target_src}/native.pdb"): 
            print(f"The file '{target_src}/native.pdb' does not exist.")
            continue

        for _, row in model_df.iterrows():
            if not os.path.exists(f"{row['model_src']}/model.pdb"): continue
            if os.path.exists(f"{row['model_src']}/scores.json"): continue

            to_run += [
                (f"{target_src}/native.pdb", row["model_src"])
            ]


        if not len(to_run): 
            print(f"Completed all models '{target_src}'")
            continue
        
        print(f"Running '{target_src}'")

        if args.procs == 1: run_ost(tqdm(to_run, desc="Running"))
        else:
            dist_factor = 0.25
            chunk_size = math.ceil( dist_factor * (len(to_run)/args.procs) )

            with tqdm(total=len(to_run),desc="Running") as pbar:

                pool = Pool(processes=args.procs)

                jobs = [
                    pool.apply_async(
                        run_ost, 
                        args=(chunk,), 
                        callback=lambda r: pbar.update(r)
                    ) 
                    for chunk in chunked(to_run, chunk_size)
                ]

                pool.close()
                pool.join()

                for j in jobs: j.get()

    ###################################

    print("#"*15)
    print("Done!")
    print("#"*15)

    ###################################
