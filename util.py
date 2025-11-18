###################################

import os
import math
import string
import numpy as np

from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import one_to_three
from Bio import pairwise2
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial.distance import cdist

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

###################################

def calc_distance(v1, v2): return sum((a-b)**2 for a, b in zip(v1, v2))**0.5

###################################

def calc_angle(v1, v2):

    mag_v1 = math.sqrt(sum(i*j for i, j in zip(v1, v1)))
    mag_v2 = math.sqrt(sum(i*j for i, j in zip(v2, v2)))

    cos_val = sum(i*j for i, j in zip(v1, v2)) / (mag_v1 * mag_v2)

    if cos_val < -1: cos_val = -1
    if cos_val > 1: cos_val = 1

    angle = math.degrees(math.acos(cos_val))

    assert angle >= 0 and angle <= 180, ("angle output is not in correct range!",angle)

    return angle

###################################

def encoder(entry, alphabet = {
    'A': 0,'C': 1,'D': 2,'E': 3,
    'F': 4,'G': 5,'H': 6,'I': 7,
    'K': 8,'L': 9,'M': 10,'N': 11,
    'P': 12,'Q': 13,'R': 14,'S': 15,
    'T': 16,'V': 17,'W': 18,'Y': 19,
    'U': 20, 'O': 21,
    'A-RNA': 22, 'U-RNA': 23,
    'C-RNA': 24, 'G-RNA': 25
}): 
    assert entry in alphabet, (entry, alphabet)

    encoded = [0]*len(alphabet)
    encoded[alphabet[entry]] = 1
  
    return encoded

###################################

def parse_pdb(pdb_file, strict = True):  

    chains = {}

    for raw_line in open(pdb_file,'r'):
        line = raw_line.strip()

        if line[:4] == "ATOM": 
            chain_id = line[21]
            resseq = int(line[22:26].strip())
            residue = line[17:20].strip()

            if residue == "UNK" or residue == "X": continue

            if chain_id not in chains: chains[chain_id] = {"res": {}}

            if resseq not in chains[chain_id]["res"]: 
                knd = None
                if len(residue) == 1: knd = "RNA" 
                elif len(residue) == 2: knd = "UNK"
                else: knd = "PROT"

                chains[chain_id]["res"][resseq] = {
                    "name": residue,
                    "kind": knd,
                    "atoms": []
                }
            
            chains[chain_id]["res"][resseq]["atoms"] += [{
                "name": line[12:16].strip(),
                "coord": [ # -> [x, y, z]
                    float(line[30:38].strip()), 
                    float(line[38:46].strip()),
                    float(line[46:54].strip())
                ]
            }]

    for chain in chains: 
        
        polymer_kind = None

        for residue in chains[chain]["res"]: 

            if polymer_kind is None: polymer_kind = chains[chain]["res"][residue]["kind"]

            assert polymer_kind == chains[chain]["res"][residue]["kind"], (polymer_kind, chains[chain]["res"][residue]["kind"])

            if chains[chain]["res"][residue]["kind"] == "PROT": 
                root = None

                for atom in range(len(chains[chain]["res"][residue]["atoms"])): 
                    if chains[chain]["res"][residue]["name"] == "GLY" and chains[chain]["res"][residue]["atoms"][atom]["name"] == "CA": 
                        root = atom

                    elif chains[chain]["res"][residue]["name"] != "GLY" and chains[chain]["res"][residue]["atoms"][atom]["name"] == "CB": 
                        root = atom

                if root is not None: 
                    chains[chain]["res"][residue]["pos"] = chains[chain]["res"][residue]["atoms"][root]["coord"]
                elif strict: 
                    assert root is not None, (root, residue, chains[chain]["res"][residue]["name"], polymer_kind, chains[chain]["res"][residue]["atoms"])
                else:
                    temp_coords = []
                    for atom in chains[chain]["res"][residue]["atoms"]: 
                        temp_coords += [atom['coord']]
                    temp_coords = np.array(temp_coords)

                    chains[chain]["res"][residue]["pos"] = np.mean(temp_coords, axis = 0).tolist()

            elif chains[chain]["res"][residue]["kind"] == "RNA": 
                root = None

                for atom in range(len(chains[chain]["res"][residue]["atoms"])): 
                    if chains[chain]["res"][residue]["atoms"][atom]["name"] == "C3'": root = atom

                if root is not None: 
                    chains[chain]["res"][residue]["pos"] = chains[chain]["res"][residue]["atoms"][root]["coord"]
                elif strict: 
                    assert root is not None, (root, residue, chains[chain]["res"][residue]["name"], polymer_kind, chains[chain]["res"][residue]["atoms"])
                else:
                    temp_coords = []
                    for atom in chains[chain]["res"][residue]["atoms"]: 
                        temp_coords += [atom['coord']]
                    temp_coords = np.array(temp_coords)

                    chains[chain]["res"][residue]["pos"] = np.mean(temp_coords, axis = 0).tolist()

        chains[chain]["resseq"] = [x for x in sorted(chains[chain]["res"])]
        chains[chain]["seq"] = [chains[chain]["res"][x]["name"] for x in chains[chain]["resseq"]]
        chains[chain]["polymer_kind"] = polymer_kind

        if polymer_kind == "PROT": chains[chain]["seq"] = "".join(map(three_to_one,chains[chain]["seq"]))
        else: chains[chain]["seq"] = "".join(chains[chain]["seq"])

        assert len(chains[chain]["seq"]) == len(chains[chain]["resseq"])

    return chains

###################################
