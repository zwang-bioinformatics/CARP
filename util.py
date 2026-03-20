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
from numba import njit
from numba.typed import List

###################################

@njit
def calc_distance(v1, v2): 
    out = 0
    for i in range(len(v1)):
        out += (v1[i] - v2[i])**2
    return out**(0.5)
    
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

def calc_com_feats(pdb_file, node_features, subunit_length):
    com_feats = {}
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)

    for model in structure:
        for chain in model:
            for residue in chain.get_unpacked_list():
                for atom in residue.get_unpacked_list():
                    if np.isnan(atom.mass): residue.detach_child(atom.id)
                if len(residue) == 0: chain.detach_child(residue.id)

    model_com = list(structure.center_of_mass())
    chain_com = {str(chain.id):list(chain.center_of_mass()) for chain in structure.get_chains()}
 
    max_chain = {}
    max_model = None

    unscaled_model = []
    unscaled_chain = []
    discounted_chain = []
    model_angle = []
    chain_angle = []

    for i, coord in enumerate(node_features["coord"]): 
        model_dist = calc_distance(coord, model_com)
        chain_dist = calc_distance(coord, chain_com[node_features["chain_id"][i]])

        if node_features["chain_id"][i] not in max_chain: max_chain[node_features["chain_id"][i]] = chain_dist
        if max_model is None: max_model = model_dist

        if chain_dist > max_chain[node_features["chain_id"][i]]: max_chain[node_features["chain_id"][i]] = chain_dist
        if model_dist > max_model: max_model = model_dist

        unscaled_model += [model_dist]
        unscaled_chain += [chain_dist]

        ci = ((subunit_length[node_features["chain_id"][i]] - 1) / 2)
        dist_ri_ci = abs(node_features["seq_idx"][i] - ci)
        discounted_chain += [chain_dist / (1 + math.log(1 + dist_ri_ci))]

        model_angle += [calc_angle(coord, model_com)]
        chain_angle += [calc_angle(coord, chain_com[node_features["chain_id"][i]])]

    scaled_model = []
    scaled_chain = []

    for i in range(len(node_features["coord"])): 
        scaled_model += [unscaled_model[i] / max_model]
        scaled_chain += [unscaled_chain[i] / max_chain[node_features["chain_id"][i]]]

    return {
        "angle_model_com": model_angle,
        "scaled_dist_model_com":  scaled_model,
        "unscaled_dist_model_com": unscaled_model,

        "angle_chain_com": chain_angle,
        "discounted_dist_chain_com":  discounted_chain,
        "scaled_dist_chain_com":  scaled_chain,
        "unscaled_dist_chain_com": unscaled_chain
    }

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

def parse_aln(alignment): 

    best = None

    for aln in sorted(alignment): # to try and make determinstic

        if best is None: best = aln
        if aln.score > best.score: best = aln

    return best

###################################

def map_aln(aln, resseq): 

    curr_seq = aln.seqA
    true_seq = aln.seqB

    aln_map = {}

    ref_idx = []
    curr_i = 0
    for i, r in enumerate(curr_seq): 
        if r != '-': ref_idx += [curr_i]; curr_i += 1
        else: ref_idx += [None]

    curr_i = 0
    for i, r in enumerate(true_seq): 
        if r == "-": continue
        if r != curr_seq[i]: aln_map[resseq[curr_i]] = None
        else: aln_map[resseq[curr_i]] = ref_idx[i]
        curr_i += 1

    return aln_map

###################################

def parse_bpseq(bpseq_fl): 
    out_map = {}
    curr_c = None
    for line in map(lambda l: l.strip(), open(bpseq_fl,'r')):
        if line[0] == "#": 
            curr_c = " ".join(line.split()[1:-2]); 
            assert curr_c not in out_map
            out_map[curr_c] = {}
            continue

        if len(line) == 0: continue
        elements = line.split()
        if len(elements) != 3: print("ISSUE!!", elements); continue
        out_map[curr_c][int(elements[0]) - 1] = {"base": elements[1], "connector" : int(elements[2]) - 1}
    return out_map

###################################

def parse_mat(mat_fl): 
    out_map = {}
    curr_c = None
    for ln in map(lambda l: l.strip(), open(mat_fl,'r')):
        if not len(ln): continue
        if ln[0] == ">": 
            if ln[1:] not in out_map: out_map[ln[1:]] = {}
            curr_c = ln[1:]
            continue
        vals = ln.split()
        out_map[curr_c][frozenset([int(vals[0]) - 1, int(vals[1]) - 1])] = float(vals[2])
    return out_map

###################################

def parse_fasta(fasta_fl, fetch_ids = False):
    seqs = []
    ids = []
    curr_seq = None
    for raw_line in open(fasta_fl,'r'): 
        line = raw_line.strip()
        if ">" in line: 
            if curr_seq is not None: seqs += [(curr_seq, set(curr_seq.replace("X","")))]; curr_seq = ""
            else: curr_seq = ""
            if fetch_ids: ids += [line.strip()]
        else: curr_seq += line
    if curr_seq is not None: seqs += [(curr_seq, set(curr_seq.replace("X","")))]
    if fetch_ids: return seqs, ids
    return seqs

###################################

def parse_rnaview(src_dir, tag="model"): 

    torsion_fl = src_dir + f"{tag}.pdb_new_torsion.out"
    if not os.path.exists(torsion_fl): return

    out_fl = src_dir + f"{tag}.pdb.out"
    if not os.path.exists(out_fl): return

    ##############

    out_map = {
        "torsions": {},
        "interacting": set(),
        "interactions": {},
    }

    ##############    

    for line in map(lambda x: x.strip(), open(torsion_fl, 'r')):
        values = line.split()

        if len(values) != 10: continue
        if values[0] == "Ch" and values[1] == "Res": continue # header

        # ['Ch', 'Res', 'Num', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
        out_map["torsions"][(values[0],int(values[2]))] = {
            key: float(values[i + 3])
            for i, key in enumerate(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi'])
        }

    ##############

    start = False
    
    for line in map(lambda x: x.strip(), open(out_fl, 'r')):

        if line == "BEGIN_base-pair": start = True; continue
        if line.strip() == "END_base-pair": break
        if not start: continue
        values = line.split()
        # print(values, len(values))

        nuc_A_chain, nuc_A_res = values[1:3]
        nuc_A_chain = nuc_A_chain[0]

        nuc_B_res, nuc_B_chain = values[4:6]
        nuc_B_chain = nuc_B_chain[0]
        
        out_map["interactions"][(nuc_A_chain, int(nuc_A_res), nuc_B_chain, int(nuc_B_res))] = {
            "syn": int("syn" in line),"stacked": int("stacked" in line),
            "cis": int("cis" in line), "trans": int("tran" in line),
            "pair": values[3]
        }

        out_map["interacting"] |= set([(nuc_A_chain, int(nuc_A_res)), (nuc_B_chain, int(nuc_B_res))])

    return out_map

###################################

def parse_forgi(src_dir, chains, tag="model"): 
    out_map = {
        "interacting": set(),
        "secondary_class": {}
    }

    for fl in os.listdir(src_dir): 
        if fl[-3:] != ".cg" or tag not in fl: continue

        seq_ids = None

        for line in map(lambda x: x.strip(), open(src_dir + fl, 'r')):

            if line[:11] == "interacting": 
                chain, resseq = line.split()[1].split(":")
                if "." in resseq: resseq = resseq.split(".")[0]

                out_map["interacting"] |= set([(chain, int(resseq))])

            if line[:7] == "seq_ids": seq_ids = line.split()[1:]
            if line[:6] == "define": 
                values = line.split()
                kind = values[1][0]

                for idx in range(int(len(values[2:])/2)):
                    for placement in range(int(values[2 + 2*idx]), int(values[2 + 2*idx + 1]) + 1): 
                        chain, resseq = seq_ids[placement - 1].split(":")
                        if "." in resseq: resseq = resseq.split(".")[0]

                        out_map["secondary_class"][
                            (
                                chain, 
                                int(resseq)
                            )
                        ] = kind

    return out_map

###################################

def parse_netsurfp(input_csv): 
    output = {}

    for li,line in enumerate(open(input_csv,"r")):
        if li == 0: continue
        #id, seq, n, rsa, asa, q3, p[q3_H], p[q3_E], p[q3_C], q8, p[q8_G], p[q8_H], p[q8_I], p[q8_B], p[q8_E], p[q8_S], p[q8_T], p[q8_C], phi, psi, disorder
        
        chain_id, resname, resseq, rsa, _, _, p_q3_H, p_q3_E, p_q3_C, _, p_q8_G, p_q8_H, p_q8_I, p_q8_B, p_q8_E, p_q8_S, p_q8_T, p_q8_C, phi, psi, disorder = line.strip().split(",")
        if chain_id.replace(">",'').strip() not in output: output[chain_id.replace(">",'').strip()] = {}

        output[chain_id.replace(">",'').strip()][int(resseq)-1] = {
            "resname": resname,
            "rsa": rsa,
            "p_q3_H": p_q3_H,
            "p_q3_E": p_q3_E,
            "p_q3_C": p_q3_C,
            "disorder": disorder,
            "phi": phi,
            "psi": psi
        }

    return output

###################################

def parse_amigos(all_sprd): 
    if not os.path.exists(all_sprd): return 
    mapper = {}
    for i, line in enumerate(map(lambda x: x.strip(), open(all_sprd, 'r'))):
        if i == 0: continue
        entries = line.split("\t")
        assert len(entries) == 14, entries
        chain_id = entries[1][0].strip()
        res_idx = entries[1][1:].strip()
        eta = float(entries[3].strip())
        theta = float(entries[4].strip())
        mapper[(chain_id, int(res_idx))] = {"eta": eta, "theta": theta}

    return mapper

###################################

@njit
def fast_edge_idx(coords, atoms):
    # probably can improve speed with a KD-Tree in future...
    
    edge_idx = List()

    for i in range(len(coords)):
        ilist = List()

        for j in range(i + 1, len(coords)): 

            bb_dist = np.sqrt(np.square(coords[i] - coords[j]).sum())
            atom_dist = -1
            # probably a faster way to do this, but about max 20 atoms per res so not worth
            for atom_i in atoms[i]:
                for atom_j in atoms[j]:
                    curr_adist = np.sqrt(np.square(atom_i - atom_j).sum())
                    if atom_dist == -1: atom_dist = curr_adist
                    if curr_adist < atom_dist: atom_dist = curr_adist

            if bb_dist <= 14 or atom_dist <= 6:
                ilist.append((j, bb_dist, atom_dist))

        edge_idx.append(ilist)

    return edge_idx

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

def load_target_features(target_src):

    assert os.path.exists(f"{target_src}/prot.fasta")
    assert os.path.exists(f"{target_src}/rna.fasta")

    prot_seq, prot_ids = parse_fasta(f"{target_src}/prot.fasta", fetch_ids = True)
    rna_seq, rna_ids = parse_fasta(f"{target_src}/rna.fasta", fetch_ids = True)

    required_files = {
        "prot_ref": [f"{target_src}/{p.replace('>', '').strip()}.pdb" for p in prot_ids], # i.e. colabfold
        "IPKnot": [f"{target_src}/out.bpseq"],
        "LinearPartition": [f"{target_src}/bp.mat"],
        "NSP3": [f"{target_src}/nsp/01/01.csv"],
    }

    ####################

    skip = False
    for tool in required_files: 
        for fl in required_files[tool]:
            if not os.path.exists(fl): 
                print(f"> !!MISSING!! {tool}: {required_files[tool]}")
                skip = True
    if skip: return

    ####################    

    nsp_data = parse_netsurfp(f"{target_src}/nsp/01/01.csv")

    if len(nsp_data) == 0: 
        print("ERROR NSP DATA NOT FOUND!")
        return
        
    ####################
    
    ipknot_data = parse_bpseq(f"{target_src}/out.bpseq")

    if len(ipknot_data) == 0: 
        print("ERROR IPKnot DATA NOT FOUND!")
        return

    ####################

    linear_partition_data = parse_mat(f"{target_src}/bp.mat")

    if len(linear_partition_data) == 0: 
        print("ERROR Linear Partition DATA NOT FOUND!")
        return

    ####################

    pdb_refs = {}
    for prot_id in prot_ids: 
        pdb_refs[prot_id.replace(">",'').strip()] = parse_pdb(target_src + "/" + prot_id.replace(">", "").strip() + ".pdb")
        if pdb_refs[prot_id.replace(">",'').strip()] is None:
            print("ERROR PDB REF NOT PARSED:", prot_id)

    if any(pdb_refs[ref] is None for ref in pdb_refs): return

    return prot_seq, prot_ids, rna_seq, rna_ids, nsp_data, ipknot_data, linear_partition_data, pdb_refs

###################################

def aggregate_features(tf, model_src):
    assert os.path.exists(f"{model_src}/model.pdb")

    prot_seq, prot_ids, rna_seq, rna_ids, nsp_data, ipknot_data, linear_partition_data, pdb_refs = tf

    required_files = {
        "DSSP": [f"{model_src}/dssp.npy"],
        "FORGI": [f"{model_src}/forgi_out/"],
        "AMIGOS": [f"{model_src}/amigos_output/all_sprd.txt"],
        "RNAView": [f"{model_src}/RNAView_out/model.pdb.out", f"{model_src}/RNAView_out/model.pdb_new_torsion.out"]
    }

    ####################

    skip = False
    for tool in required_files: 
        for fl in required_files[tool]:
            if not os.path.exists(fl): 
                print(f"> !!MISSING!! {tool}: {required_files[tool]}")
                skip = True
    if skip: return

    if not any([".cg" in fl for fl in os.listdir(f"{model_src}/forgi_out/")]): return

    #################### 

    parsed_pdb = parse_pdb(f"{model_src}/model.pdb", strict = False)

    if parsed_pdb is None: print("ERROR: unable to parse model"); return

    rnaview = parse_rnaview(f"{model_src}/RNAView_out/")
    amigos_torsions = parse_amigos(f"{model_src}/amigos_output/all_sprd.txt")
    forgi = parse_forgi(f"{model_src}/forgi_out/", [chain for chain in parsed_pdb if parsed_pdb[chain]["polymer_kind"] == "RNA"])

    if rnaview is None or forgi is None or amigos_torsions is None: 
        print(f"ERROR: RNA model tool failed [RNAView:{rnaview is None}, forgi:{forgi is None}, AMIGOS: {amigos_torsions is None}]")        
        return    

    ####################

    try: dssp = np.load(f"{model_src}/dssp.npy",allow_pickle=True).item()
    except: print("ERROR LOADING DSSP FEATURES"); return

    ####################

    netsurfp_keys = ['rsa', 'p_q3_H', 'p_q3_E', 'p_q3_C', 'disorder', 'psi', 'phi']
    dssp_keys = {
        "secondary_structure": 2, "rsa": 3,
        "phi": 4, "psi": 5
    }

    node_features = {
        "res_idx": [],
        "seq_idx": [],
        "chain_idx": [],
        "chain_id": [],
        "subunit_id": [],
        "sequence": [],
        "coord": [],
        "raw_sequence": [],
        "polymer_kind": [],
        "interacting_ipknot": [],
        "interacting_forgi": [],
        "interacting_rnaview": [],
        "prot_rna_interface": [],
        "amigos_eta": [],
        "amigos_theta": []
    }
    
    for key in dssp_keys: node_features[key + '_model_dssp'] = []
    for key in netsurfp_keys: node_features[key + "_netsurfp"] = []

    edge_features = {
        "model_atom_distance": [],
        "model_bb_distance": [],
        "model_angle": [],
        "is_inter_chain": [],
        "is_prot_rna": [],

        "ref_distance": [], # colabfold
        "ref_angle": [], # colabfold

        "rnaview_pair": [],
        "ipknot_pair": [],
        "linear_partition_prob": []
    }

    edge_index = []

    ###################

    subunit_length = {}

    chain2idx = {}
    rna_seqs = [seq[0] for seq in rna_seq]
    prot_seqs = [seq[0] for seq in prot_seq]

    assert len(set(rna_seqs)) == len(rna_seqs)
    assert len(set(prot_seqs)) == len(prot_seqs)

    for chain in sorted(parsed_pdb): 

        if chain not in chain2idx: chain2idx[chain] = len(chain2idx)

        ###################

        best = None
        subunit_seq = None
        curr_seq = parsed_pdb[chain]["seq"]

        if parsed_pdb[chain]["polymer_kind"] == "RNA": 
            if curr_seq in rna_seqs: 
                fid = rna_ids[rna_seqs.index(curr_seq)][1:]
                subunit_seq = curr_seq
            else:
                for s_idx, poss_seq in enumerate(rna_seqs):
                    aln = parse_aln(pairwise2.align.globalxx(poss_seq, curr_seq, penalize_end_gaps=(True, False)))
                    if best is None: best = aln; subunit_seq = poss_seq; fid = rna_ids[s_idx][1:]
                    if aln.score > best.score: best = aln; subunit_seq = poss_seq; fid = rna_ids[s_idx][1:]

        elif parsed_pdb[chain]["polymer_kind"] == "PROT":
            if curr_seq in prot_seqs: 
                fid = prot_ids[prot_seqs.index(curr_seq)][1:]
                subunit_seq = curr_seq
            else:
                for s_idx, poss_seq in enumerate(prot_seqs):
                    aln = parse_aln(pairwise2.align.globalxx(poss_seq, curr_seq, penalize_end_gaps=(True, False)))
                    if best is None: best = aln; subunit_seq = poss_seq; fid = prot_ids[s_idx][1:]
                    if aln.score > best.score: best = aln; subunit_seq = poss_seq; fid = prot_ids[s_idx][1:]

        else: assert False, "ERROR UNSUPPORTED POLYMER KIND...."
        
        best = parse_aln(pairwise2.align.globalxx(subunit_seq, curr_seq, penalize_end_gaps=(True, False)))
        aln_map = map_aln(best, parsed_pdb[chain]["resseq"])
        subunit_length[chain] = len(subunit_seq)

        ###################

        for i, res_idx in enumerate(parsed_pdb[chain]["resseq"]): 
            if res_idx not in aln_map or aln_map[res_idx] == None: continue 

            ###################

            if parsed_pdb[chain]["polymer_kind"] == "PROT": 
                if (chain, res_idx) not in dssp: continue

                node_features['subunit_id'] += [fid]
                node_features['polymer_kind'] += [0]
                node_features['sequence'] += [encoder(curr_seq[i])]
                for key in netsurfp_keys: 
                    node_features[key + "_netsurfp"] += [
                        float(nsp_data[fid][aln_map[res_idx]][key])
                    ]

                for key in dssp_keys: 
                    if key == "secondary_structure": node_features[key + '_model_dssp'] += [encoder(
                        dssp[(chain, res_idx)][dssp_keys[key]],
                        alphabet={
                            "H": 0, "B": 1, "E": 2, "G": 3,
                            "I": 4, "T": 5, "S": 6, "-": 7
                        }
                    )]
                    else: node_features[key + '_model_dssp'] += [dssp[(chain, res_idx)][dssp_keys[key]]]

                node_features['interacting_ipknot'] += [-1]
                node_features["interacting_rnaview"] += [-1]

                node_features["interacting_forgi"] += [-1]
                node_features["amigos_eta"] += [9999.9999]
                node_features["amigos_theta"] += [9999.9999]

            ###################

            elif parsed_pdb[chain]["polymer_kind"] == "RNA":
                if curr_seq[i] not in ["A", "C", "G", "U"]: continue

                node_features['subunit_id'] += [fid]
                node_features['polymer_kind'] += [1]
                node_features['sequence'] += [encoder(curr_seq[i] + "-RNA")]
                for key in netsurfp_keys: node_features[key + "_netsurfp"] += [-1]

                for key in dssp_keys: 
                    if key == "secondary_structure": node_features[key + '_model_dssp'] += [[-1]*8]
                    else: node_features[key + '_model_dssp'] += [-1]

                node_features['interacting_ipknot'] += [
                    int(ipknot_data[fid][aln_map[res_idx]]['connector'] != -1)
                ]

                node_features["interacting_rnaview"] += [int((chain, res_idx) in rnaview["interacting"])]
                node_features["interacting_forgi"] += [int((chain, res_idx) in forgi["interacting"])]    

                if (chain, res_idx) in amigos_torsions: 
                    node_features["amigos_eta"] += [amigos_torsions[(chain, res_idx)]["eta"]]
                    node_features["amigos_theta"] += [amigos_torsions[(chain, res_idx)]["theta"]]
                else: 
                    node_features["amigos_eta"] += [9999.9999]
                    node_features["amigos_theta"] += [9999.9999]
            
            ###################

            else: continue

            ###################

            node_features['raw_sequence'] += [curr_seq[i]]
            node_features['res_idx'] += [res_idx]
            node_features['seq_idx'] += [aln_map[res_idx]]
            node_features['chain_id'] += [chain]
            node_features['chain_idx'] += [chain2idx[chain]]
            node_features['coord'] += [parsed_pdb[chain]['res'][res_idx]['pos']]

            assert curr_seq[i] == subunit_seq[aln_map[res_idx]], (src_dir + pdb, chain, res_idx, subunit_seq[aln_map[res_idx]])

            ###################

    com_feats = calc_com_feats(f"{model_src}/model.pdb", node_features, subunit_length)
    for key in com_feats: node_features[key] = com_feats[key]

    ###################

    atoms = []

    for i, coord_i in enumerate(node_features["coord"]): 

        ###################

        i_atoms = np.array([atom["coord"] for atom in parsed_pdb[node_features["chain_id"][i]]["res"][node_features["res_idx"][i]]["atoms"]])
        atoms += [i_atoms]

    edge_idx_hits = fast_edge_idx(np.array(node_features["coord"]), atoms)

    ###################

    for i, coord_i in enumerate(node_features["coord"]): 

        ###################

        i_atoms = np.array([atom["coord"] for atom in parsed_pdb[node_features["chain_id"][i]]["res"][node_features["res_idx"][i]]["atoms"]])
        i_kind = parsed_pdb[node_features["chain_id"][i]]["polymer_kind"] 
        i_subunit = node_features["subunit_id"][i]
        i_chain = node_features["chain_id"][i]
        i_res_idx = node_features["res_idx"][i]

        prot_rna_interface = False

        ###################
        
        for hit in edge_idx_hits[i]:
            j = hit[0]

            coord_j = node_features["coord"][j]

            ###################

            j_kind = parsed_pdb[node_features["chain_id"][j]]["polymer_kind"]
            j_subunit  = node_features["subunit_id"][j]
            j_chain = node_features["chain_id"][j]
            j_res_idx = node_features["res_idx"][j]

            ###################

            bb_dist = hit[1]
            atom_dist = hit[2]

            assert atom_dist != -1
    
            if atom_dist <= 6 and j_kind != i_kind: prot_rna_interface = True
            
            ###################
            
            if bb_dist <= 14 or atom_dist <= 6:                       
            
                edge_features["is_prot_rna"] += [int((i_kind == "RNA" and j_kind == "PROT") or (j_kind == "RNA" and i_kind == "PROT"))]*2
                edge_features["is_inter_chain"] += [int(node_features["chain_id"][i] != node_features["chain_id"][j])]*2
                edge_features["model_atom_distance"] += [atom_dist]*2
                edge_features["model_bb_distance"] += [bb_dist]*2
                edge_features["model_angle"] += [calc_angle(coord_i, coord_j)]*2

                ###################

                if j_kind == "PROT" and i_kind == "PROT" and node_features["chain_id"][i] == node_features["chain_id"][j]: 
                    try: 
                        i_alpha_coord = pdb_refs[i_subunit]["A"]["res"][1 + node_features['seq_idx'][i]]["pos"]
                        j_alpha_coord = pdb_refs[j_subunit]["A"]["res"][1 + node_features['seq_idx'][j]]["pos"]
                        edge_features["ref_distance"] += [calc_distance(i_alpha_coord, j_alpha_coord)]*2
                        edge_features["ref_angle"] += [calc_angle(i_alpha_coord, j_alpha_coord)]*2
                    except: 
                        assert False, (i_subunit, j_subunit, 1 + node_features['seq_idx'][i], 1 + node_features['seq_idx'][j])
                else: 
                    edge_features["ref_distance"] += [-1]*2
                    edge_features["ref_angle"] += [-1]*2

                ###################

                if j_kind == "RNA" and i_kind == "RNA" and node_features["chain_id"][i] == node_features["chain_id"][j]: 
                    seq_idx_i,seq_idx_j = node_features['seq_idx'][i], node_features['seq_idx'][j]
                    pair_idx = frozenset([seq_idx_i, seq_idx_j])
                    ij = (i_chain,i_res_idx,j_chain, j_res_idx)
                    ji = (j_chain,j_res_idx,i_chain, i_res_idx)

                    linear_partition_prob = 0
                    if pair_idx in linear_partition_data[i_subunit]: linear_partition_prob = linear_partition_data[i_subunit][pair_idx]
                    edge_features["ipknot_pair"] += [int(ipknot_data[i_subunit][seq_idx_i]["connector"] == seq_idx_j or ipknot_data[j_subunit][seq_idx_j]["connector"] == seq_idx_i)]*2
                    edge_features["linear_partition_prob"] += [linear_partition_prob]*2
                    edge_features["rnaview_pair"] += [int(ij in rnaview["interactions"] or ji in rnaview["interactions"])]*2
                else: 
                    edge_features["ipknot_pair"] += [-1]*2
                    edge_features["linear_partition_prob"] += [-1]*2
                    edge_features["rnaview_pair"] += [-1]*2

                ###################

                edge_index += [[i,j], [j,i]]

                ###################

        ###################
                        
        node_features["prot_rna_interface"] += [int(prot_rna_interface)]

    ###################

    return node_features, edge_features, edge_index, chain2idx

    ###################

###################################

