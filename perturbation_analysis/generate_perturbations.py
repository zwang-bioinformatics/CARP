###################################

# Author: Andrew Jordan Siciliano

###################################

# Environment: 
#   > conda activate PyRosetta
#   > export LD_PRELOAD=/home/asiciliano/anaconda3/envs/PyRosetta/lib/libtcmalloc.so:$LD_PRELOAD

###################################

import os
import sys
sys.path.append(f"../")

from init import *
sys.path.append(f"{ROOT}/src/")

from tqdm import tqdm

###################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dataset", 
    action="store", type=str, default = "DOCKING",
)

args = parser.parse_args()

###################################

tasks = {}

if args.dataset in ["DOCKING"]: 
    src_dir = f"{ROOT}/data/targets/{args.dataset}/"

    for target in os.listdir(src_dir):
        target_src = f"{src_dir}/{target}/"

        if not all([
            (
                # See: https://colab.research.google.com/drive/1XQ4-WRKAOXa81XZbcEosrG7dS09gc8MI?usp=sharing
                os.path.exists(target_src + "/nsp/01/01.csv") or os.path.exists(target_src + "/nsp/results.csv") # server output?
            ), 
            os.path.exists(target_src + "bp.mat")
        ]): continue

        tasks[target_src] = []

        for model in os.listdir(f"{target_src}/models/"):
            model_src = f"{target_src}/models/{model}/"

            ##################
            
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

            ##################
            
            if all(checks): 
                assert os.path.exists(model_src + "model.pdb")
                tasks[target_src] += [model_src]

            ##################

else:
    assert False, "NOT IMPLEMENTED YET!"


###################################

import json
import math
import argparse
from termcolor import colored
from multiprocessing import Pool
from more_itertools import chunked
import shutil
import Bio
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import is_nucleic

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore')#, PDBConstructionWarning)

###################################

print("#"*30)

from pyrosetta import *
from pyrosetta.rosetta import *
from rosetta.core.fragment import * # https://github.com/Reproducibility-FBB-MSU/Ab-Initio-Protein-Structure-Prediction/blob/master/ab-initio-protein-structure-prediction.ipynb
from pyrosetta.rosetta.protocols.simple_moves import SmallMover, ShearMover
import pyrosetta.rosetta.protocols.rigid as rigid_moves
import pyrosetta.rosetta.protocols.docking as docking

init("-ignore_zero_occupancy false", set_logging_handler=True)

logger = logging.getLogger("rosetta")
logger.setLevel(logging.CRITICAL)

import pyrosetta.distributed

pyrosetta.distributed.init()

###################################
# nohup python -u generate_perturbations.py -d DOCKING > dock_merge.log &

def perturb_batch(batch, repeats = 2): 

    for task in batch: 

        ################

        in_pdb, out_src = task

        ################

        if False: # CHANGE TO CHECK FOR WHETHER COMPLETED...
            try: 
            # if True:
                parser = PDBParser()
                structure = parser.get_structure("struct", in_pdb)

                model_blank = Bio.PDB.StructureBuilder.StructureBuilder()
                model_blank.init_structure("")
                model_blank.init_model(model_id="1")
                model_blank = list(model_blank.structure)[0]

                rna_structure = model_blank.copy()
                protein_structure = model_blank.copy()

                for chain in structure.get_chains():
                    chain.detach_parent()
                    contains_rna = False
                    for residue in chain: 
                        if is_nucleic(residue): contains_rna = True; break
                    if contains_rna: rna_structure.add(chain)
                    else: protein_structure.add(chain)

                if not os.path.exists(out_src + "rna_chains.pdb"): 
                    io = PDBIO()
                    io.set_structure(rna_structure)
                    io.save(out_src + "in_rna_chains.pdb")

                if not os.path.exists(out_src + "protein_chains.pdb"): 
                    io = PDBIO()
                    io.set_structure(protein_structure)
                    io.save(out_src + "in_protein_chains.pdb")

                ################################
                
                movemap = MoveMap()
                movemap.set_bb(True) 
                
                SHEAR_0 = ShearMover(movemap, 1.0, 25)
                SHEAR_0.angle_max(35)
                
                SHEAR_1 = ShearMover(movemap, 1.0, 75)
                SHEAR_1.angle_max(60)

                SMALL = SmallMover(movemap, 1.0, 150)
                SMALL.angle_max(10)

                protein_state = pose_from_pdb(out_src + "in_protein_chains.pdb")

                ################

                for i in range(repeats): 

                    task_src = out_src + f"tasks/prot_{i}/"
                    if os.path.exists(task_src + 'p_info.txt'): continue
                    perturbated_state = Pose()
                    perturbated_state.assign(protein_state)

                    if not os.path.exists(task_src): os.makedirs(task_src)

                    ################

                    SHEAR_1.apply(perturbated_state)

                    ################

                    perturbated_state.dump_pdb(task_src + "prot_raw.pdb")
                    with open(task_src + 'p_info.txt','w') as fl: fl.write("SHEAR_1")

                ################

                for i in range(repeats, 2*repeats): 

                    task_src = out_src + f"tasks/prot_{i}/"
                    if os.path.exists(task_src + 'p_info.txt'): continue
                    
                    perturbated_state = Pose()
                    perturbated_state.assign(protein_state)

                    if not os.path.exists(task_src): os.makedirs(task_src)

                    ################

                    SMALL.apply(perturbated_state)
                    SHEAR_0.apply(perturbated_state)

                    ################

                    perturbated_state.dump_pdb(task_src + "prot_raw.pdb")
                    with open(task_src + 'p_info.txt','w') as fl: fl.write("SMALL|SHEAR_0")

                ################################

                rna_state = pose_from_pdb(out_src + "in_rna_chains.pdb")
                rna_fragset = core.import_pose.libraries.RNA_LibraryManager.get_instance().rna_fragment_library(
                    "/home/asiciliano/anaconda3/envs/PyRosetta/lib/python3.9/site-packages/pyrosetta/database/sampling/rna/RNA18_HUB_2.154_2.5.torsions"
                )
                # https://www.zora.uzh.ch/id/eprint/199191/1/fastef_thesis.pdf -> FARFAR2, draws on an updated fragment library (RNA18 HUB 2.154 2.5.torsions and RNA18 HUB 2.154 2.5.jump, [156, 361]

                ################

                for i in range(repeats): 

                    task_src = out_src + f"tasks/rna_{i}/"
                    if os.path.exists(task_src + 'p_info.txt'): continue

                    perturbated_state = Pose()
                    perturbated_state.assign(rna_state)

                    ################

                    if not os.path.exists(task_src): os.makedirs(task_src)

                    ################

                    atom_level_domain_map = core.pose.toolbox.AtomLevelDomainMap(perturbated_state)
                    frag_mover = protocols.rna.denovo.movers.RNA_FragmentMover(rna_fragset, atom_level_domain_map, 1, 0)
                    frag_mover.random_fragment_insertion(perturbated_state, 1)

                    ################

                    perturbated_state.dump_pdb(task_src + "rna_raw.pdb")
                    with open(task_src + 'p_info.txt','w') as fl: fl.write("RNA_FRAGMOVER(1-mer)")

                for i in range(repeats, 2*repeats): 

                    task_src = out_src + f"tasks/rna_{i}/"
                    if os.path.exists(task_src + 'p_info.txt'): continue

                    perturbated_state = Pose()
                    perturbated_state.assign(rna_state)

                    ################

                    if not os.path.exists(task_src): os.makedirs(task_src)

                    ################

                    atom_level_domain_map = core.pose.toolbox.AtomLevelDomainMap(perturbated_state)
                    frag_mover = protocols.rna.denovo.movers.RNA_FragmentMover(rna_fragset, atom_level_domain_map, 1, 0)
                    frag_mover.random_fragment_insertion(perturbated_state, 3)

                    ################

                    perturbated_state.dump_pdb(task_src + "rna_raw.pdb")
                    with open(task_src + 'p_info.txt','w') as fl: fl.write("RNA_FRAGMOVER(3-mer)")

                ################################

            except: 
                print("FAILED:", task)
                continue

        protein = {}
        rna = {}

        if not os.path.exists(out_src + "merged/"): os.makedirs(out_src + "merged/")

        for task_name in os.listdir(out_src + "tasks/"): 
            if task_name[:3] == "rna": rna[task_name] = "\n".join(ln.strip() for ln in open(out_src + "tasks/" + task_name + "/rna_raw.pdb",'r') if ln[:4] == "ATOM")
            elif task_name[:4] == "prot": protein[task_name] = "\n".join(ln.strip() for ln in open(out_src + "tasks/" + task_name + "/prot_raw.pdb",'r') if ln[:4] == "ATOM")

        rna["rna_original"] = "\n".join(ln.strip() for ln in open(out_src + "in_rna_chains.pdb",'r') if ln[:4] == "ATOM")
        protein["prot_original"] = "\n".join(ln.strip() for ln in open(out_src + "in_protein_chains.pdb",'r') if ln[:4] == "ATOM")

        for prot_pdb in protein: 
            for rna_pdb in rna: 
                if "original" in prot_pdb and "original" in rna_pdb: continue # this is essentially no change.... skip it
                output_name = prot_pdb + "." + rna_pdb
                if not os.path.exists(out_src + "merged/" + output_name + "/"): os.makedirs(out_src + "merged/" + output_name)

                if not os.path.exists(out_src + "merged/" + output_name + "/model.pdb"): 
                    with open(out_src + "merged/" + output_name + "/model.pdb", 'w') as fl: fl.write("\n".join([protein[prot_pdb],rna[rna_pdb]]))

    return len(batch)

###################################

for task in tasks:
    for model_src in tqdm(tasks[task], desc = f"Perturbing {task}"):
        os.makedirs(model_src + "/raw_perturb/", exist_ok=True)
        perturb_batch([
            (model_src + "model.pdb", model_src + "/raw_perturb/")
        ])

###################################
