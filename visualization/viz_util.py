###################################

import warnings; warnings.filterwarnings("ignore") # not good in practice but fine for our purposes

import pandas as pd
from tqdm import tqdm
import os
import argparse

import json
import pandas as pd

import sys
sys.path.append(f"../")

from init import *
from util import *

from multiprocessing import Pool
from more_itertools import chunked

from Bio.PDB import PDBParser, is_aa
from numba import njit

###################################

def format_title(title, bold = False):
    out = r"$\bf{" if bold else r"$"
    if "CARP" in title: 
        out += title.replace(":", u"\u2010") + r"^{*}"
    elif "TM" in title:
        out += "AlphaFold3" + u"\u2010" + title
    else: 
        out += title.replace("-", u"\u2010")
    out += r"}$" if bold else r"$"

    return out

###################################

def mm_scale(df):
    mn, mx = df.min(), df.max()
    if mn == 0 and mx == 0: return df
    return (2*(df - mn) / (mx - mn)) - 1

###################################

@njit
def compute_af3_rscore(chain_idx, polymer_kind, pae):
    out = np.zeros(2)
    # note since it is protiein-rna complex all frames are valid...
    #   > see /home/asiciliano/alphafold3/src/alphafold3/model/features.py, sets mask to true anytime for prot/rna
    # https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

    N = len(pae)
    d_0 = 1.24 * (
        max([N, 19]) - 15
    )**(1/3) - 1.8

    for align_res in range(N):
        align_ptm = 0
        align_iptm = 0
        inum = 0

        align_iptm_rp = 0
        inum_rp = 0

        for i in range(N):            
            dn = 1 + (pae[align_res,i]/d_0)**2
            align_ptm += (1/dn)

            if chain_idx[align_res] != chain_idx[i]:
                align_iptm += (1/dn)
                inum += 1

        align_ptm /= pae.shape[1]
        align_iptm /= inum

        if align_ptm > out[0]: 
            out[0] = align_ptm

        if align_iptm > out[1]: 
            out[1] = align_iptm

    return out

###################################

def gather(sub_tasks, include_other=True):
    out = {
        "global": [],
        # , model, num_rna_chains, num_prot_chains, tm, oligo_gdtts, bb_lddt, ics, ips, ilddt, rmsd, CARP:bb_lddt,CARP:oligo_gdtts,CARP:oligo_gdtha,CARP:ilddt,CARP:ics,CARP:ips,CARP:Merged,CARP:Iface,CARP:Fold,CARP:RP,FTDMP,DARS-RNP,QUASI-RNP,DRPScore,ITScore-PR,3dRPC

        "interface": [],
        # ,model,num_rna_chains,num_prot_chains,interface,CARP:PerIface,CARP:PerIface-G,FTDMP,DARS-RNP,QUASI-RNP,DRPScore,ITScore-PR,3dRPC,ics

    }
    for tsk in sub_tasks:
        target, target_src, model_src = tsk

        ###################

        rna_chains, prot_chains = [], []

        for md in PDBParser().get_structure("struct", model_src + "model.pdb"): # MAYBE DO SOMETHING FASTER LATER....
            for chain in md:
                if any(is_aa(r) for r in chain): prot_chains += [chain.get_id()]
                else: rna_chains += [chain.get_id()]
        
        ###################

        prot_chains = "".join(prot_chains)
        rna_chains = "".join(rna_chains)

        global_quality = {
            "target": target,
            "target_src": target_src,
            "model": model_src,
            "num_rna_chains": len(rna_chains),
            "num_prot_chains": len(prot_chains),
        }

        ###################

        for q in ["tm", "oligo_gdtts", "bb_lddt", "ics", "ips", "ilddt", "rmsd"]:
            global_quality[q] = None

        true_scores = None
        if os.path.exists(model_src + "scores.json"):
            true_scores = json.load(open(model_src + "/scores.json",'r'))
            global_quality["rmsd"] = true_scores["rmsd"]
            global_quality['irmsd_protein_fit'] = true_scores["irmsd_protein_fit"]
            global_quality['irmsd_interface_fit'] = true_scores["irmsd_interface_fit"]

            for q in ["tm", "oligo_gdtts", "bb_lddt", "ics", "ips", "ilddt", "qs_best"]:
                global_quality[q] = max([true_scores[q] for unit in true_scores])

        if os.path.exists(model_src + "rmsd_3dRPC.json"):
            rpcrmsd = json.load(open(model_src + "/rmsd_3dRPC.json",'r'))
            global_quality["irmsd"] = rpcrmsd["I_rmsd"]

        ###################

        casp_scores = None
        if os.path.exists(model_src + "casp_scores.json"): # will overwrite whatever is in the scores.json if it exists... takes precedence
            casp_scores = json.load(open(model_src + "/casp_scores.json",'r'))
            for key in casp_scores: 
                if casp_scores[key] == -1: casp_scores[key] = None
            
            global_quality["tm"] = casp_scores['TMscore']
            global_quality["bb_lddt"] = casp_scores['lDDT']
            global_quality["ilddt"] = casp_scores["ilDDT"]
            global_quality["oligo_gdtts"] = casp_scores["GDT_TS"]
            global_quality["rmsd"] = casp_scores["RMSD"]
            global_quality["ics"] = casp_scores["ICS(F1)"]
            global_quality["ips"] = casp_scores["IPS"]


        ###################

        for cscore in [
                "bb_lddt", "oligo_gdtts", "oligo_gdtha", # global fold
                "ilddt", "ics", "ips", "Merged", # global interface
                "Iface", "Fold", "RP" # combined scores
        ]: global_quality[f"CARP:{cscore}"] = None

        ###################

        for oscore in [
            "FTDMP","DARS-RNP","QUASI-RNP",
            "DRPScore","ITScore-PR","3dRPC"
        ]: global_quality[oscore] = None

        if "af3_output" in model_src:
            jtag = model_src.split("results")[1].split("seed")[0][1:-1]
            mtag = "seed" + model_src.split("seed")[1]

            if not os.path.exists(model_src + "tm_preds.npy"):
                confidence = f"/home/asiciliano/CARP/data/targets/DOCKING/{target}/af3_output/results/"
                confidence = f"{confidence}/{jtag}/{mtag}/{jtag}_{mtag}_confidences.json"
                assert os.path.exists(confidence), confidence
                confidence = json.load(open(confidence,'r'))
                pt = compute_af3_rscore(np.array(confidence["token_chain_ids"]), np.array(confidence["pae"]))
                np.save(model_src + "tm_preds.npy", pt)
            else: 
                pt = np.load(model_src + "tm_preds.npy")

            global_quality["pTM"] = pt[0]
            global_quality["ipTM"] = pt[1]
        else: 
            global_quality["ipTM"] = None
            global_quality["pTM"] = None

        ###################

        carp = None
        if os.path.exists(model_src + "/predicted_quality/carp.pkl"):
            carp = pd.read_pickle(model_src + "/predicted_quality/carp.pkl")
            # print(carp)
            carp = carp[carp['config'].str.contains('fold')] # kill old models...
            carp["Iface"] = (carp["ics"] + carp["ips"] + carp["ilddt"] ) / 3
            carp["Fold"] = (carp["bb_lddt"] + carp["oligo_gdtts"] + carp["oligo_gdtha"] ) / 3
            carp["Merged"] = (carp["Fold"] + carp["Iface"]) / 2
            # carp['Merged'] = carp[['Fold', 'Iface']].apply(hmean, axis=1)

            carp["RP"] = carp.apply(
                lambda r: None
                    if not len(r["interface"]) else (
                        (r["interface_ics_pred"] + r["interface_ips_pred"])/2
                    ).mean(), 
                axis=1
            )

            for cscore, score in carp[
                [
                    "bb_lddt", "oligo_gdtts", "oligo_gdtha", 
                    "ilddt", "ics", "ips", 
                    "Iface","Fold","Merged","RP"
                ]
            ].mean().to_dict().items(): global_quality[f"CARP:{cscore}"] = score
            
        # if carp is None:
        #     print("MISSING CARP!", model_src )
    
        ###################

        drp_scores = None
        try: 
            if include_other:
                drp_scores = fetch_drpscores(model_src + "/predicted_quality/")
                if drp_scores is not None: 
                    global_quality["DRPScore"] = drp_scores["overall_interface"]

        except: drp_scores = None
        if drp_scores is None and include_other:
            print("MISSING DRP.... ->",model_src)
            
        ##################

        rpc_scores = None
        try: 
            if include_other:
                rpc_scores = fetch_3dRPC(model_src + "/predicted_quality/")
                if rpc_scores is not None:
                    global_quality["3dRPC"] = rpc_scores["3dRPC_full"]
                    # preds["3dRPC_iface"] = rpc_scores["3dRPC_iface"]
        except: rpc_scores = None

        ##################

        if include_other:
            global_quality["ITScore-PR"] = fetch_itscorepr(model_src + "/predicted_quality/")
            global_quality["FTDMP"] = parse_ftdmp(model_src + "/predicted_quality/")
            global_quality["DARS-RNP"] = fetch_darsrnp(model_src + "/predicted_quality/")
            global_quality["QUASI-RNP"] = fetch_quasirnp(model_src + "/predicted_quality/")

        # print(global_quality, os.listdir(model_src + "/predicted_quality/"))

        out["global"] += [global_quality]

        ###################

        interfaces = {}
        for _, row in carp.iterrows():
            row = row.to_dict()
            for i, dimer in enumerate(row["interface"]):
                iface = ".".join(sorted(dimer))
                if iface not in interfaces: interfaces[iface] = {
                    "target": target,
                    "target_src": target_src,
                    "model": model_src, 
                    "num_rna_chains": len(rna_chains), 
                    "num_prot_chains": len(prot_chains), 
                    "interface": iface,
                    "reference_interface": None,
                    "CARP:PerIface": []
                }
                interfaces[iface]["CARP:PerIface"] += [(row["interface_ips_pred"][i] + row["interface_ics_pred"][i]) / 2]
        for iface in interfaces: 
            interfaces[iface]["CARP:PerIface"] = np.mean(interfaces[iface]["CARP:PerIface"])
            interfaces[iface]["CARP:PerIface-G"] = (interfaces[iface]["CARP:PerIface"] + global_quality["CARP:Iface"]) / 2

        if include_other:
            if not os.path.exists(model_src + "/predicted_quality/dimer_scores/"): continue

            for dimer in os.listdir(model_src + "/predicted_quality/dimer_scores/"):
                iface = ".".join(sorted(dimer.split("_")))
                if iface not in interfaces: interfaces[iface] = {
                    "target": target,
                    "target_src": target_src,
                    "model": model_src, 
                    "num_rna_chains": len(rna_chains), 
                    "num_prot_chains": len(prot_chains), 
                    "interface": iface,
                    "reference_interface": None
                }      

                try: interfaces[iface]["FTDMP"] = parse_ftdmp(model_src + f"/predicted_quality/dimer_scores/{dimer}/")
                except: interfaces[iface]["FTDMP"] = None

                try: interfaces[iface]["DARS-RNP"] = fetch_darsrnp(model_src + f"/predicted_quality/dimer_scores/{dimer}/")
                except: interfaces[iface]["DARS-RNP"] = None

                try: interfaces[iface]["QUASI-RNP"] = fetch_quasirnp(model_src + f"/predicted_quality/dimer_scores/{dimer}/")
                except: interfaces[iface]["QUASI-RNP"] = None

                try: interfaces[iface]["ITScore-PR"] = fetch_itscorepr(model_src + f"/predicted_quality/dimer_scores/{dimer}/")
                except: interfaces[iface]["ITScore-PR"] = None

            if drp_scores is not None and drp_scores["interface_preds"] is not None:
                for dimer in drp_scores["interface_preds"]:
                    iface = ".".join(sorted(dimer))
                    if iface not in interfaces: interfaces[iface] = {
                        "target": target,
                        "target_src": target_src,
                        "model": model_src, 
                        "num_rna_chains": len(rna_chains), 
                        "num_prot_chains": len(prot_chains), 
                        "interface": iface,
                        "reference_interface": None
                    }      
                    interfaces[iface]["DRPScore"] = drp_scores["interface_preds"][dimer]
            
            if rpc_scores is not None and rpc_scores["3dRPC_iface"] is not None:
                for ca, cb, sc in rpc_scores["3dRPC_iface"]:
                    iface = ".".join(sorted([ca, cb]))
                    if iface not in interfaces: interfaces[iface] = {
                        "target": target,
                        "target_src": target_src,
                        "model": model_src, 
                        "num_rna_chains": len(rna_chains), 
                        "num_prot_chains": len(prot_chains), 
                        "interface": iface,
                        "reference_interface": None
                    }      
                    interfaces[iface]["3dRPC"] = sc

        if true_scores is not None:

            for i,dimer in enumerate(true_scores["contact_target_interfaces"]):
                if dimer[0] not in true_scores["chain_mapping"] or dimer[1] not in true_scores["chain_mapping"]: continue

                dimer = [true_scores["chain_mapping"][dimer[0]], true_scores["chain_mapping"][dimer[1]]]
                iface = ".".join(sorted(dimer))

                if iface not in interfaces: continue # no predictors predicted this interface, ground truth will never be used...

                if len(true_scores["per_interface_ics"]): 
                    if true_scores["per_interface_ics"][i] == -1: interfaces[iface]["ics"] = None
                    else: interfaces[iface]["ics"] = true_scores["per_interface_ics"][i]
                else: interfaces[iface]["ics"] = None

                if len(true_scores["per_interface_ips"]): 
                    if true_scores["per_interface_ips"][i] == -1: interfaces[iface]["ips"] = None
                    else: interfaces[iface]["ips"] = true_scores["per_interface_ips"][i]
                else: interfaces[iface]["ips"] = None

        for iface in interfaces:
            if "ics" not in interfaces[iface]: interfaces[iface]["ics"] = None
            if "ips" not in interfaces[iface]: interfaces[iface]["ips"] = None

        if true_scores is not None:
            chain_map = {mdl_ch: trg_ch for trg_ch, mdl_ch in true_scores["chain_mapping"].items()}

            for iface in interfaces.keys():
                a,b = iface.split(".")
                if a not in chain_map or b not in chain_map: continue
                # interfaces["model_interface"]
                interfaces[iface]["reference_interface"] = ".".join(sorted(
                    [chain_map[a], chain_map[b]]
                ))

        # chain mapping here is not so clear... using scores.json always for interface scores
        # if casp_scores is not None:
        #     for i, dimer in enumerate(casp_scores["ICS_Interfaces"]):
        #         iface = ".".join(sorted(dimer))
        #         if iface not in interfaces: continue # no predictors predicted this interface, ground truth will never be used...

        #         if casp_scores["ICS_perInterface"][i] == -1: interfaces[iface]["ics"] = None
        #         else: interfaces[iface]["ics"] = casp_scores["ICS_perInterface"][i]
        #     for iface in interfaces:
        #         if "ics" not in interfaces[iface]: interfaces[iface]["ics"] = None

        for iface in interfaces:
            for key in [
                "CARP:PerIface","CARP:PerIface-G",
                "FTDMP","DARS-RNP","QUASI-RNP","DRPScore",
                "ITScore-PR","3dRPC", "ics", "ips"
            ]:
                if key not in interfaces[iface]: interfaces[iface][key] = None

            out["interface"] += [interfaces[iface]]

    return out, len(sub_tasks)

###################################
