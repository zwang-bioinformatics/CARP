###################################

# Author: Andrew Jordan Siciliano

###################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:1024,expandable_segments:False"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,expandable_segments:True"

import warnings; warnings.filterwarnings("ignore") # not good in practice but fine for our purposes

import copy
import sys
sys.path.append(f"../")

from tqdm import tqdm

import argparse

import json
import pandas as pd

from init import *
from util import *

sys.path.append(f"{ROOT}/src/")

import torch
torch.set_num_threads(100)

from torch_geometric.data import Dataset, Data

from run_util import *

from model import *

from multiprocessing import Pool
from more_itertools import chunked

###################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dataset", 
    action="store", type=str, default = "CASP16",
)

args = parser.parse_args()

###################################

device = "cuda:0" #"cuda:0"
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

global_keys = ["bb_lddt", "oligo_gdtts", "oligo_gdtha", "ilddt", "ics", "ips"]

###################################

node_abilation_groups = [
    # [ [feat keys], polymer_kind ]

    # PROTEIN -> 
    
    [['sequence'], 0],
    [['scaled_dist_model_com', 'unscaled_dist_chain_com'], 0],
    [['angle_model_com', 'angle_chain_com'], 0],

    [['rsa_model_dssp'], 0],
    [['secondary_structure_model_dssp', 'psi_model_dssp', 'phi_model_dssp'], 0],

    [['rsa_netsurfp'], 0],
    [['disorder_netsurfp'], 0],
    [['p_q3_H_netsurfp', 'p_q3_E_netsurfp', 'p_q3_C_netsurfp', 'psi_netsurfp', 'phi_netsurfp'], 0],

    # RNA ->
    [['sequence'], 1],
    [['scaled_dist_model_com', 'unscaled_dist_chain_com'], 1],
    [['angle_model_com', 'angle_chain_com'], 1],

    [['interacting_ipknot'], 1],
    [['interacting_forgi', 'interacting_rnaview'], 1],
    [['amigos_eta', 'amigos_theta',], 1],
]

###################################

edge_abilation_groups = [
    # [ [feat keys], polymer_kind ]

    [['ref_distance'], 0],  # PROT-PROT
    [['ref_angle'], 0],  # PROT-PROT

    [['model_atom_distance', 'model_bb_distance'], 0],  # PROT-PROT
    [['model_atom_distance', 'model_bb_distance'], 1], # RNA-RNA
    [['model_atom_distance', 'model_bb_distance'], 2], # prot-RNA

    [['model_angle'], 0], # PROT-PROT
    [['model_angle'], 1], # RNA-RNA
    [['model_angle'], 2], # prot-RNA

    [['rnaview_pair'], 1], # RNA-RNA
    [['linear_partition_prob'], 1],  # RNA-RNA
    [['ipknot_pair'], 1] # RNA-RNA
]

###################################

def run_model(batch, sidx = None):
    with torch.no_grad():

        out_preds = []
        for model_tag in predictors: 
            interface_out, global_out = predictors[model_tag]["model"](
                norm_feats(
                    batch["node_features"].clone(), 
                    predictors[model_tag]["norm_stats"], 
                    batch["out_idx_node"][0][0],
                    nf_keys,
                ),
                norm_feats(
                    batch["edge_features"].clone(), 
                    predictors[model_tag]["norm_stats"], 
                    batch["out_idx_edge"][0][0],
                    ef_keys,
                ),
                batch["edge_index"].clone().permute(1,0).long(),#.to(device),
                batch["interface_connect"].clone().long(),#.to(device),
                torch.unsqueeze(batch["polymer_kind"].clone().float(), dim=1),#.to(device),
                torch.unsqueeze(batch["prot_rna_interface"].clone().float(), dim=1),#.to(device),
                batch["topk_idx"].clone().long(),#.to(device),
                batch["batch"].clone().long(),#.to(device)
            )
            # global_out = torch.squeeze(sigmoid(global_out)).cpu().numpy()#.item()
            global_out = sigmoid(global_out).cpu().numpy()#.item()
            
            for batch_idx in range(len(batch["src_dir"])):

                predictions = {
                    "interface": [],
                    "interface_ics_pred": [],
                    "interface_ips_pred": []
                }

                batch_iface = None
                if interface_out is not None: 
                    iface_out_bi = sigmoid(interface_out[batch["interface_batch"] == batch_idx]).cpu().numpy()
                    for i, iface_idx in enumerate(batch["interface_map"][batch_idx]): 
                        interface_ics_pred = iface_out_bi[i][0]
                        interface_ips_pred = iface_out_bi[i][1]
                        predictions["interface"] += [batch["interface_names"][batch_idx][0][i]] # extra index in collatte (thats why [0])
                        predictions["interface_ics_pred"] += [interface_ics_pred]
                        predictions["interface_ips_pred"] += [interface_ips_pred]

                    predictions["interface_ics_pred"] = np.array(predictions["interface_ics_pred"])
                    predictions["interface_ips_pred"] = np.array(predictions["interface_ips_pred"])

                #################
                
                for i, key in enumerate(global_keys): 
                    predictions[key] = global_out[batch_idx,i]#.item() 
                
                predictions["src_dir"] = batch["src_dir"][batch_idx][0] # extra index in collatte (thats why [0])
                predictions["config"] = model_tag

                if sidx is not None:
                    predictions["sample"] = f"perm:{sidx[batch_idx]}"

                out_preds += [predictions]

        return out_preds


###################################

batch_size = 8
num_samples = 32 if args.dataset == "CASP16" else 8

for target in os.listdir(f"/home/asiciliano/CARP/data/targets/{args.dataset}/"):
    target_src = f"/home/asiciliano/CARP/data/targets/{args.dataset}/{target}/"

    if not os.path.exists(target_src + "/carp_abilation.pkl"):

        prev = None

        df = []

        with torch.no_grad():

            sigmoid = nn.Sigmoid()

            for model in tqdm(list(os.listdir(f"{target_src}/models/")), desc = f"running target: {target}"):

                model_src = f"{target_src}/models/{model}/"
                agged_features = np.load(model_src + "/agged_features.npy", allow_pickle=True)
                example = build_features(agged_features, {})
                example["src_dir"] = [model_src]
                example = example.to(device)
                batch = collate_fn([example]).to(device)
                base_predictions = run_model(batch)

                ##################################
                
                for features, polymer_kind in node_abilation_groups:

                    node_idx = (example["polymer_kind"] == polymer_kind).nonzero(as_tuple=True)[0]

                    for base in base_predictions:
                        base["sample"] = "base"
                        base["size"] = node_idx.shape[0]
                        base["features"] = features + [['PROT','RNA'][polymer_kind]]
                        base["polymer_kind"] = ['PROT','RNA'][polymer_kind]
                        df += [copy.deepcopy(base)]

                    #################

                    batch = []
                    sidx = []
                    fidx_list = []
                    for feat in features: fidx_list += list(range(example["out_idx_node"][0][feat][0],example["out_idx_node"][0][feat][1]))
                    fidx_list = torch.tensor(fidx_list).to(device)

                    for sample in range(num_samples):
                        sub = example.clone()
                        perm = torch.randperm(n=node_idx.shape[0])
                        sub["node_features"][node_idx[:,None], fidx_list] = sub["node_features"][
                            node_idx[perm][:,None], fidx_list
                        ]
                        batch += [sub]
                        sidx += [sample]

                        if (len(batch) % batch_size == 0) or (sample == num_samples - 1):
                            for out in run_model(collate_fn(batch).to(device), sidx):
                                out["features"] = features + [['PROT','RNA'][polymer_kind]]
                                out["polymer_kind"] = ['PROT','RNA'][polymer_kind]
                                out["size"] = node_idx.shape[0]
                                df += [copy.deepcopy(out)]
                                # print(out)
                            batch = []
                            sidx = []
                            
                ##################################

                for features, polymer_kind in edge_abilation_groups:

                    edge_mask = example["polymer_kind"][example["edge_index"]].sum(dim=1)
                    edge_mask[edge_mask > 0] = 3 - edge_mask[edge_mask > 0]
                    edge_mask = (edge_mask == polymer_kind).nonzero(as_tuple=True)[0]
                    
                    for base in base_predictions:
                        base["sample"] = "base"
                        base["size"] = edge_mask.shape[0]
                        base["features"] = features + [['PROT-PROT','RNA-RNA', 'PROT-RNA'][polymer_kind]]
                        base["polymer_kind"] = ['PROT-PROT','RNA-RNA', 'PROT-RNA'][polymer_kind]
                        
                        df += [copy.deepcopy(base)]

                    #################

                    batch = []

                    edge_mask = example["polymer_kind"][example["edge_index"]].sum(dim=1)
                    edge_mask[edge_mask > 0] = 3 - edge_mask[edge_mask > 0]
                    edge_mask = (edge_mask == polymer_kind).nonzero(as_tuple=True)[0]

                    batch = []
                    sidx = []
                    fidx_list = []
                    for feat in features: fidx_list += list(range(example["out_idx_edge"][0][feat][0],example["out_idx_edge"][0][feat][1]))
                    fidx_list = torch.tensor(fidx_list).to(device)

                    for sample in range(num_samples):
                        sub = example.clone()
                        perm = torch.randperm(n=edge_mask.shape[0])
                        sub["edge_features"][edge_mask[:,None], fidx_list] = sub["edge_features"][
                            edge_mask[perm][:,None], fidx_list
                        ]

                        batch += [sub]
                        sidx += [sample]

                        if (len(batch) % batch_size == 0) or (sample == num_samples - 1):
                            for out in run_model(collate_fn(batch).to(device), sidx):
                                out["features"] = features + [['PROT-PROT','RNA-RNA', 'PROT-RNA'][polymer_kind]]
                                out["polymer_kind"] = ['PROT-PROT','RNA-RNA', 'PROT-RNA'][polymer_kind]
                                out["size"] = edge_mask.shape[0]
                                df += [copy.deepcopy(out)]
                            batch = []
                            sidx = []

        df = pd.DataFrame(df)
        df.to_pickle(target_src + "/carp_abilation.pkl")
        print(df)

###################################