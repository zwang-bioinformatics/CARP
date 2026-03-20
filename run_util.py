###################################

import json
import copy
import torch
from pprint import pprint
from torch_geometric.data import Dataset, Data
# torch.set_num_threads(64)
from init import *

###################################

def norm_feats(feats, norms, dims, feat_keys):

    with torch.no_grad():
        out = feats.clone()#torch.empty_like(feats)
        for feat_key in feat_keys:
            # out[:, dims[feat_key][0]:dims[feat_key][1]] = feats[:, dims[feat_key][0]:dims[feat_key][1]]

            if feat_key in norms:
                msk = None
                if feat_key in ["ref_distance", "linear_partition_prob"]: msk = out[:, dims[feat_key][0]:dims[feat_key][1]] < 0
                out[:, dims[feat_key][0]:dims[feat_key][1]] = torch.tanh(
                    (out[:, dims[feat_key][0]:dims[feat_key][1]] - norms[feat_key]["mu"])/ norms[feat_key]["sigma"]
                )
                if msk is not None: 
                    out[:, dims[feat_key][0]:dims[feat_key][1]] = (out[:, dims[feat_key][0]:dims[feat_key][1]] + 1) / 2
                    out[:, dims[feat_key][0]:dims[feat_key][1]][msk] = -1
        return out

###################################

def format_angles(in_angles): 

    rad_tensor = torch.unsqueeze(torch.deg2rad(in_angles),dim=1)

    assert len(rad_tensor.shape) == 2

    return (torch.cat((torch.cos(rad_tensor),torch.sin(rad_tensor)),dim=1) + 1) / 2

###################################

def build_features(agged_features, norm_stats):

    with torch.no_grad():

        # node_features, edge_features, edge_index, chain2idx = agged_features

        nf, ef, edge_index, chain2idx = agged_features
    
        node_features = {}
        edge_features = {}
        # print(nf["polymer_kind"])
        # print(nf["raw_sequence"])
        for f in nf:
            if f not in ['chain_id', 'subunit_id', 'raw_sequence']:
                node_features[f] = torch.tensor(nf[f])
            else: node_features[f] = copy.deepcopy(nf[f])
            # else: node_features[f] = nf[f]

        for f in ef:
            edge_features[f] = torch.tensor(ef[f])

        #### Node Features ####
        
        out_idx_node = {}

        example = Data()

        # chain_idx, chain_id

        example["prot_rna_interface"] = node_features["prot_rna_interface"]
        example["polymer_kind"] = node_features["polymer_kind"]
        example["topk_idx"] = node_features["chain_idx"]
        example["chain_idx"] = node_features["chain_idx"]

        nidx = 0
        for nfeat_key in nf_keys:

            # nfeat = raw_example.get_tensor(nfeat_key)
            nfeat = node_features[nfeat_key]
            
            if nfeat_key in norm_stats: 
                assert norm_stats[nfeat_key]["sigma"] != 0, ("ZERO STANDARD DEVIATION!",nfeat_key)
                nfeat = torch.tanh((nfeat - norm_stats[nfeat_key]["mu"]) / norm_stats[nfeat_key]["sigma"])
                
            elif nfeat_key in ANGLE_FEATS: 
                mask = None

                if nfeat_key in ["amigos_eta", "amigos_theta"]: mask = nfeat != 9999.9999
                elif nfeat_key not in ["angle_chain_com", "angle_model_com"]: mask = example["polymer_kind"] != 0 # not PROT
                
                nfeat = format_angles(nfeat)
                if mask is not None: nfeat[mask] = -1

            if len(nfeat.shape) == 1: nfeat = torch.unsqueeze(nfeat,dim=1)
            if "node_features" not in example: example["node_features"] = nfeat
            else: example["node_features"] = torch.cat((example["node_features"],nfeat),dim=1)
            
            out_idx_node[nfeat_key] = [nidx, nidx + nfeat.shape[1]]

            # print(nfeat_key, nidx, torch.all(nfeat == example["node_features"][:, nidx:(nidx + nfeat.shape[1])]), example["node_features"].shape,"\n")
            nidx += nfeat.shape[1]

        #### Edge Features ####

        out_idx_edge = {}

        example["edge_index"] = torch.tensor(edge_index)

        assert example["edge_index"].min() >= 0 and example["edge_index"].max() < nfeat.shape[0], ("edge index issue.... ???", idx)

        eidx = 0
        for efeat_key in ef_keys: 

            efeat = edge_features[efeat_key]

            if efeat_key in norm_stats: 
                assert norm_stats[efeat_key]["sigma"] != 0, ("ZERO STANDARD DEVIATION!",efeat_key)
                msk = None
                if efeat_key in ["ref_distance", "linear_partition_prob"]: msk = efeat < 0
                efeat = torch.tanh((efeat - norm_stats[efeat_key]["mu"]) / norm_stats[efeat_key]["sigma"])
                if msk is not None: 
                    efeat = (efeat + 1) / 2
                    efeat[msk] = -1

            elif efeat_key in ANGLE_FEATS: 
                mask = efeat == -1
                efeat = format_angles(efeat)
                efeat[mask] = -1

            if len(efeat.shape) == 1: efeat = torch.unsqueeze(efeat,dim=1)
            if "edge_features" not in example: example["edge_features"] = efeat
            else: example["edge_features"] = torch.cat((example["edge_features"],efeat),dim=1)

            out_idx_edge[efeat_key] = [eidx, eidx + efeat.shape[1]]
            # print(efeat_key, eidx, torch.all(efeat == example["edge_features"][:, eidx:(eidx + efeat.shape[1])]), example["edge_features"].shape,"\n")
            eidx += efeat.shape[1]

        interface_names = []
        interface_connect = None
        interface_map = []
        # print(example["polymer_kind"])
        chains = list(sorted(chain2idx))
        for ai, A in enumerate(chains): 
            for B in chains[ai:]: 
                if A == B: continue
                curr_connect = ((example['chain_idx'] == chain2idx[A]) | (example['chain_idx'] == chain2idx[B])).nonzero()
                # print(A, B, len(meta["chain2idx"]), example["chain_idx"].shape, example["chain_idx"].min(), example["chain_idx"].max(), example["topk_idx"].min(), example["topk_idx"].max())

                
                if curr_connect.shape[0] == 0: continue
                chk = example["polymer_kind"][curr_connect].sum()

                # print(A, B, (chk == 0 or chk == example["polymer_kind"][curr_connect].shape[0]) == (not example["polymer_kind"][curr_connect][[0,-1]].sum() == 1))
                # print("CHK",chk, example["polymer_kind"][curr_connect])
                if chk == 0 or chk == example["polymer_kind"][curr_connect].shape[0]: continue
                    
                curr_connect = torch.cat(
                    (
                        curr_connect, 
                        torch.tensor([len(interface_names)]).repeat(curr_connect.shape[0],1)
                    ), dim=1
                )
                interface_names += [list(sorted([A,B]))]
                if interface_connect is None: interface_connect = curr_connect
                else: interface_connect = torch.cat((interface_connect, curr_connect), dim=0)
                interface_map += [len(interface_map)]

        # print(interface_map)
        
        example["interface_map"] = torch.tensor(interface_map)
        example["interface_connect"] = interface_connect if interface_connect is not None else torch.tensor([])
        example['num_chains'] = torch.tensor(len(chain2idx))
        example["interface_names"] = [interface_names]
        example["out_idx_node"] = [out_idx_node]
        example["out_idx_edge"] = [out_idx_edge]

        return example

###################################

def collate_fn(batch): 

    # gc.collect()

    data = Data()

    curr_interface = 0
    curr_node = 0
    curr_chain = 0

    for batch_idx, graph in enumerate(batch):

        for key in graph.keys(): 

            in_ft = graph[key]

            if torch.is_tensor(in_ft) and len(in_ft.shape) == 0: in_ft = torch.unsqueeze(in_ft, dim=0)

            if key in ["interface_map", "src_dir", "pdb_id", "idx", 'interface_names', "out_idx_node", "out_idx_edge"]: 
                if key not in data: data[key] = [in_ft]
                else: data[key] += [in_ft]
                continue
            
            if key == "interface_connect" and len(in_ft.shape) == 2: 
                in_ft[:, 0] += curr_node
                in_ft[:, 1] += curr_interface

            if key == "edge_index": in_ft += curr_node
            if key == "topk_idx": in_ft += curr_chain
            # print(key)
            if key not in data: data[key] = in_ft
            else: data[key] = torch.cat((data[key], in_ft), dim = 0)

        batch = torch.zeros(graph["node_features"].shape[0]) + batch_idx
        interface_batch = torch.zeros(graph["interface_map"].shape[0]) + batch_idx

        if "batch" not in data: data["batch"] = batch
        else:  data["batch"] = torch.cat((data["batch"], batch), dim = 0)

        if "interface_batch" not in data: data["interface_batch"] = interface_batch
        else: data["interface_batch"] = torch.cat((data["interface_batch"], interface_batch), dim = 0)

        curr_interface += graph["interface_map"].shape[0]
        curr_node += graph["node_features"].shape[0]
        curr_chain += graph["num_chains"]

    data["chain_idx"] = data["topk_idx"].clone()

    if data["prot_rna_interface"].sum() > 0: 
        data["topk_idx"][data["prot_rna_interface"].bool()] = torch.arange(curr_chain, curr_chain + data["prot_rna_interface"].sum()).to(data["prot_rna_interface"].device)

    data["unique_interface"] = torch.unique(data["interface_batch"].long())
    data["cnt"] = torch.zeros((len(data["src_dir"]), 2))
    data["cnt"][:, 0] = 1
    data["cnt"][data["unique_interface"], 1] = 1
    data["cnt"] = torch.sum(data["cnt"], dim=1)

    return data

###################################
