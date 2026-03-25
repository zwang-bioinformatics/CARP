###################################

# Author: Andrew Jordan Siciliano

###################################

import warnings; warnings.filterwarnings("ignore") # not good in practice but fine for our purposes

from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:1024,expandable_segments:False"
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_SHM_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'
    

import argparse

import json
import pandas as pd

from init import *
from util import *

import torch
torch.set_num_threads(128)

from torch_geometric.data import Dataset, Data

from run_util import *

import sys
sys.path.append(f"{ROOT}/src/")
from model import *

from multiprocessing import Pool
from more_itertools import chunked

###################################

def check_complete(model_src):
    checks = [
        os.path.exists(model_src + "dssp.npy"),
        os.path.exists(f"{model_src}/amigos_output/all_sprd.txt"),
    ]

    if os.path.exists(f"{model_src}/forgi_out/"):
        completed = {
            "forgi": False,
            # "element_string": False
        }
        for fl in os.listdir(f"{model_src}/forgi_out/"): 
            if ".cg" in fl: completed["forgi"] = True
            # if ".element_string" in fl: completed["element_string"] = True

        checks += [all(case for case in completed.values())]
    
    checks += [all([
        os.path.exists(f"{model_src}/RNAView_out/{fl}") 
        for fl in ["model.pdb_new_torsion.out","model.pdb.out"]
    ])]

    return all(checks)
    
###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-target_src", action="store", type=str, help="")
    parser.add_argument("-model_src", action="store", type=str, help="")
    parser.add_argument("-device", action="store", type=str, default="cpu", help = "")

    args = parser.parse_args()
    # device = "cuda:0" 
    device = args.device #"cpu"

    ###################################

    predictors = {}
    with torch.no_grad():
        for tag in model_setup:
            assert os.path.exists(tag), "ERROR: config file does not exist!"
            config = json.load(open(tag,'r'))

            weights = model_setup[tag] + "/model.pt"
            assert os.path.exists(weights), "ERROR: MODEL WEIGHTS NOT FOUND!"
            weights = torch.load(weights, map_location=device)

            predictors[tag.split("/")[-1].split(".")[0]] = {
                "model" : CARP(
                    node_dim = 66, 
                    edge_dim = 12,
                    config = config["architecture"]
                ).to(device),
                "norm_stats": json.load(open(model_setup[tag] + "/norm_stats.json",'r'))
            }

            for key in predictors[tag.split("/")[-1].split(".")[0]]["norm_stats"]:
                predictors[tag.split("/")[-1].split(".")[0]]["norm_stats"][key]["mu"] = torch.tensor(
                    predictors[tag.split("/")[-1].split(".")[0]]["norm_stats"][key]["mu"]
                ).to(device)
                predictors[tag.split("/")[-1].split(".")[0]]["norm_stats"][key]["sigma"] = torch.tensor(
                    predictors[tag.split("/")[-1].split(".")[0]]["norm_stats"][key]["sigma"]
                ).to(device)

            predictors[tag.split("/")[-1].split(".")[0]]["model"].load_state_dict(weights)
            predictors[tag.split("/")[-1].split(".")[0]]["model"].eval()

    ###################################

    global_keys = ["bb_lddt", "oligo_gdtts", "oligo_gdtha", "ilddt", "ics", "ips"]
    target_src = args.target_src
    model_src = args.model_src

    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        
        tf = load_target_features(target_src)
        assert tf is not None, "ERROR: target features not found..."
        
        ##################

        if not os.path.exists(model_src + "/agged_features.npy"):
            agged_features = aggregate_features(tf, model_src)
            np.save(model_src + "/agged_features.npy", np.array(agged_features, dtype=object))

        agged_features = np.load(model_src + "/agged_features.npy", allow_pickle=True)
        example = build_features(agged_features, {})

        example["src_dir"] = model_src
        example = collate_fn([example]).to(device)

        ##################

        for model_tag in predictors: 

            interface_out, global_out = predictors[model_tag]["model"](
                norm_feats(
                    example["node_features"].clone() , 
                    predictors[model_tag]["norm_stats"], 
                    example["out_idx_node"][0][0], # should be the same for every graph in the batch....
                    nf_keys,
                ),
                norm_feats(
                    example["edge_features"].clone(), 
                    predictors[model_tag]["norm_stats"], 
                    example["out_idx_edge"][0][0], # should be the same for every graph in the batch....
                    ef_keys,
                ),
                example["edge_index"].clone().permute(1,0).long(),#.to(device),
                example["interface_connect"].clone().long(),#.to(device),
                torch.unsqueeze(example["polymer_kind"].clone().float(), dim=1),#.to(device),
                torch.unsqueeze(example["prot_rna_interface"].clone().float(), dim=1),#.to(device),
                example["topk_idx"].clone().long(),#.to(device),
                example["batch"].clone().long(),#.to(device)
            )

            # print(interface_out)

            predictions = {
                "interface": [],
                "interface_ics_pred": [],
                "interface_ips_pred": []
            }

            #################

            if interface_out is not None: 

                interface_out = sigmoid(interface_out).cpu().numpy()

                # print(example["interface_names"][0][0])
                # print(example["interface_map"][0])
                for i, iface_idx in enumerate(example["interface_map"][0]): 
                    interface_ics_pred = interface_out[i][0]#.item()
                    interface_ips_pred = interface_out[i][1]#.item()
                    # predictions["interface"] += [meta["contact_model_interfaces"][str(iface_idx.item())]]
                    predictions["interface"] += [example["interface_names"][0][0][i]] # only one example in each batch...
                    predictions["interface_ics_pred"] += [interface_ics_pred]
                    predictions["interface_ips_pred"] += [interface_ips_pred]

                predictions["interface_ics_pred"] = np.array(predictions["interface_ics_pred"])
                predictions["interface_ips_pred"] = np.array(predictions["interface_ips_pred"])

            #################

            global_out = torch.squeeze(sigmoid(global_out)).cpu().numpy()#.item()

            for i, key in enumerate(global_keys): 
                predictions[key] = global_out[i]#.item()
            
            predictions["src_dir"] = example["src_dir"][0]
            predictions["config"] = model_tag
            df += [predictions]

        ##################

        df = pd.DataFrame(df)
        os.makedirs(model_src + "/predicted_quality/", exist_ok = True)
        df[global_keys + ["config"]].to_csv(model_src + "/predicted_quality/carp.csv")
        df.to_pickle(model_src + "/predicted_quality/carp.pkl")

###################################
