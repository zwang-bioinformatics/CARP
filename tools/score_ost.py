###################################

import sys
import json
import numpy as np
from ost import mol, io
from ost.mol import alg
from ost.seq.alg import SequenceIdentity
from ost.mol.alg import scoring, chain_mapping
from ost.mol.alg.scoring import _GetAlignedResidues
from ost.mol.alg.chain_mapping import _GetAlnPropsOne
from pprint import pprint
from ost import seq

###################################

# ~ AJ OVERWROTE THIS GUY TO FIX AN RNA EDGE CASE....
def updated_IndexMapping(self, ch1, ch2):
    """ Fetches aln and returns indices of aligned residues

    returns 2 numpy arrays containing the indices of residues in
    ch1 and ch2 which are aligned
    """

    mapped_indices_1 = list()
    mapped_indices_2 = list()
    idx_1 = 0
    idx_2 = 0
    for col in self.alns[(ch1, ch2)]:
        if col[0] != '-' and col[1] != '-':
            mapped_indices_1.append(idx_1)
            mapped_indices_2.append(idx_2)
        if col[0] != '-':
            idx_1 +=1
        if col[1] != '-':
            idx_2 +=1

    return (np.array(mapped_indices_1).astype(int), np.array(mapped_indices_2).astype(int))

ost.mol.alg.bb_lddt.BBlDDTScorer._IndexMapping = updated_IndexMapping

###################################

def updated_NWAlign(self, s1, s2, stype):
    if stype == mol.ChemType.AMINOACIDS:
        return seq.alg.SemiGlobalAlign(s1, s2, self.pep_subst_mat,
                                        gap_open=self.pep_gap_open,
                                        gap_ext=self.pep_gap_ext)[0]
    elif stype == mol.ChemType.NUCLEOTIDES:
        # ~ AJ ADDID THIS CASE...
        if s1.string == s2.string:  # fixes that weird case where low entropy all matching but bad alignment.. this is bandaid...
            aln = seq.CreateAlignment()
            aln.AddSequence(s1)
            aln.AddSequence(s2)
            return aln
        elif set(s1.string) == set(s2.string) and len(set(s1.string)) == 1:
            assert False, "ERROR BAD ALIGN (structure align later?)"
        # ~ end of AJ edits
        return seq.alg.SemiGlobalAlign(s1, s2, self.nuc_subst_mat,
                                        gap_open=self.nuc_gap_open,
                                        gap_ext=self.nuc_gap_ext)[0]
    else:
        raise RuntimeError("Invalid ChemType")
    return aln

chain_mapping.ChainMapper.NWAlign = updated_NWAlign

###################################

output_json = sys.argv[3]
reference = io.LoadPDB(sys.argv[2], no_hetatms=True)
model = io.LoadPDB(sys.argv[1], no_hetatms=True)

###################################

output = {}

s = scoring.Scorer(
    model, reference, 
    resnum_alignments=False, 
    lddt_no_stereochecks=True,
)

##################

chain_map = s.mapping.GetFlatMapping()
trg_pep_chains = [c.GetName() for c in s.chain_mapper.polypep_seqs]
trg_nuc_chains = [c.GetName() for c in s.chain_mapper.polynuc_seqs]

print("\nTarget Protein Chains:", trg_pep_chains)
print("Target Nucleotide Chains:", trg_nuc_chains)

assert len(trg_pep_chains) and len(trg_nuc_chains), "Missing protein or nucleotide chains"

##################

output = {
    "model": sys.argv[1], "reference": sys.argv[2],
    "chain_mapping": chain_map,
    "reference_chain_types": {
        chain.name: str(chain.type).replace('CHAINTYPE_', '').lower()
        for chain in reference.chains
    },
    "model_chain_types": {
        chain.name: str(chain.type).replace('CHAINTYPE_', '').lower()
        for chain in model.chains
    },
    "tm": s.tm_score, "ics": s.ics, "ips": s.ips,
    "lddt": s.lddt, "bb_lddt": s.bb_lddt, "ilddt": s.ilddt,
    "oligo_gdtts": s.gdtts, "oligo_gdtha": s.gdtha, "qs_global": s.qs_global,
    "qs_best": s.qs_best, "per_interface_ics": s.per_interface_ics,
    "per_interface_ips": s.per_interface_ips, "contact_target_interfaces": s.contact_target_interfaces,
    "contact_model_interfaces": s.contact_model_interfaces, "dockq_interfaces": s.dockq_interfaces,
    "dockq_target_interfaces": s.dockq_target_interfaces, "dockq_irmsd": s.irmsd,
    "dockq_lrmsd": s.lrmsd, "dockq_scores": s.dockq_scores,
    "dockq_fnat": s.fnat, "dockq_fnonnat": s.fnonnat,
    "dockq_wave": s.dockq_wave, "dockq_ave": s.dockq_ave,
    "rmsd": s.rmsd
}

##################

print("\n--- Sequence Alignment Summary ---")
for (t_chain, m_chain), aln in s.mapping.alns.items():
    sid = SequenceIdentity(aln)
    is_rna = s.target.Select(f"cname='{t_chain}' and aname=\"C4'\"").atom_count > 0
    type_str = "RNA" if is_rna else "Protein"
    
    print(f"Mapping {type_str}: {t_chain} <-> {m_chain}")
    print(f"  - Sequence Identity: {sid:.4f}")
    print(f"  - Alignment Length: {aln.GetLength()}\n")

    # print(aln)
    print(".......")
    
##################

interface_query = "(peptide=True and 10.0 <> [nucleotide=True]) or (nucleotide=True and 10.0 <> [peptide=True])"
reference_interface = s.target.Select(interface_query, mol.MATCH_RESIDUES)

##################

full_ref_bb = s.target.CreateEmptyView()
full_mdl_bb = s.model.CreateEmptyView()
ref_view_interface = s.target.CreateEmptyView()
mdl_view_interface = s.model.CreateEmptyView()

##################

# See _extract_mapped_pos_full_bb
#   > https://git.scicore.unibas.ch/schwede/openstructure/-/blob/master/modules/mol/alg/pymod/scoring.py?ref_type=heads#L30

for trg_ch, mdl_ch in chain_map.items():
    aln = s.mapping.alns[(trg_ch, mdl_ch)]

    if trg_ch in trg_pep_chains:
        exp_atoms = ["CA"]
    elif trg_ch in trg_nuc_chains:
        exp_atoms = ["C4'"]
    else:
        # this should be guaranteed by the chain mapper
        raise RuntimeError(f"Unexpected error (chain={trg_ch})")

    for trg_res, mdl_res in _GetAlignedResidues(aln, s.mapping.target, s.mapping.model):
        for aname in exp_atoms:
            trg_at = trg_res.FindAtom(aname)
            mdl_at = mdl_res.FindAtom(aname)
            if not (trg_at.IsValid() and mdl_at.IsValid()):
                # this should be guaranteed by the chain mapper
                raise RuntimeError(f"Unexpected error - contact OST developer ({trg_at}, {mdl_at})")
            
            # below is for view adding instead of Vec3List ~ AJ
            full_ref_bb.AddAtom(trg_at)
            full_mdl_bb.AddAtom(mdl_at)
            if reference_interface.ViewForHandle(trg_res.handle).IsValid():
                ref_view_interface.AddAtom(trg_at)
                mdl_view_interface.AddAtom(mdl_at)

##################

assert mdl_view_interface.atom_count == ref_view_interface.atom_count

ref_prot = full_ref_bb.Select('peptide=True')
mdl_prot = full_mdl_bb.Select('peptide=True')

ref_rna = full_ref_bb.Select('nucleotide=True')
mdl_rna = full_mdl_bb.Select('nucleotide=True')

##################

print()
print(f"Cleaned Reference Chains: {[c.name for c in s.target.chains]}")
print(f"Cleaned Model Chains: {[c.name for c in s.model.chains]}")

bb_model = s.model.Select('aname=CA or aname="C4\'"')
bb_ref = s.target.Select('aname=CA or aname="C4\'"')
print(f"Model Cleaned | Atoms (CA/C4'): {bb_model.atom_count} | Residues: {bb_model.residue_count} | RNA: {bb_model.Select('nucleotide=True').residue_count}")
print(f"Reference Cleaned | Atoms (CA/C4'): {bb_ref.atom_count} | Residues: {bb_ref.residue_count} | RNA: {bb_ref.Select('nucleotide=True').residue_count}\n")

assert full_mdl_bb.atom_count == full_ref_bb.atom_count

print(f"Model Global Fit Atoms (Full): {full_mdl_bb.atom_count}")
print(f"Reference Global Fit Atoms (Full): {full_ref_bb.atom_count}\n")
output["num_mapped_atoms"] = full_ref_bb.atom_count

assert mdl_prot.atom_count == ref_prot.atom_count

print(f"Model Global Fit Atoms (Protein): {mdl_prot.atom_count}")
print(f"Reference Global Fit Atoms (Protein): {ref_prot.atom_count}\n")
output["num_mapped_prot_atoms"] = ref_prot.atom_count

assert mdl_rna.atom_count == ref_rna.atom_count
print(f"Model Global Fit Atoms (RNA): {mdl_rna.atom_count}")
print(f"Reference Global Fit Atoms (RNA): {ref_rna.atom_count}\n")
output["num_mapped_rna_atoms"] = ref_rna.atom_count

print(f"Model Interface Atoms (CA/C4'): {mdl_view_interface.atom_count}")
print(f"Reference Interface Atoms (CA/C4'): {ref_view_interface.atom_count}<->{reference_interface.atom_count}\n")

assert mdl_view_interface.atom_count == ref_view_interface.atom_count

output["num_mapped_interface_atoms"] = ref_view_interface.atom_count
output["num_mapped_rna_interface_atoms"] = ref_view_interface.Select('nucleotide=True').atom_count
output["num_mapped_prot_interface_atoms"] = ref_view_interface.Select('peptide=True').atom_count
output["num_reference_interface_atoms"] = reference_interface.atom_count

if reference_interface.atom_count == 0:
    output["mapping_coverage_interface"] = None
else:
    output["mapping_coverage_interface"] = ref_view_interface.atom_count / reference_interface.atom_count

##################

if mdl_rna.atom_count == 0: 
    print("skipping...., no mapped RNA")
    print("#"*15)
    assert False, "INVESTIGATE"

if mdl_prot.atom_count == 0: 
    print("skipping...., no mapped protein")
    print("#"*15)
    assert False, "INVESTIGATE"

##################

assert full_mdl_bb.handle == mdl_view_interface.handle
assert full_ref_bb.handle == ref_view_interface.handle

alg.SuperposeSVD(full_mdl_bb, full_ref_bb, apply_transform=True)

if reference_interface.atom_count == 0:
    output["irmsd_global_fit"] = None
else:
    output["irmsd_global_fit"] = float(alg.CalculateRMSD(mdl_view_interface, ref_view_interface))

assert mdl_prot.handle == mdl_view_interface.handle
assert ref_prot.handle == ref_view_interface.handle

alg.SuperposeSVD(mdl_prot, ref_prot, apply_transform=True)
output["rmsd_prot"] = float(alg.CalculateRMSD(mdl_prot, ref_prot))

if reference_interface.atom_count == 0:
    output["irmsd_protein_fit"] = None
else:
    output["irmsd_protein_fit"] = float(alg.CalculateRMSD(mdl_view_interface, ref_view_interface))


assert mdl_rna.handle == mdl_view_interface.handle
assert ref_rna.handle == ref_view_interface.handle

alg.SuperposeSVD(mdl_rna, ref_rna, apply_transform=True)
output["rmsd_rna"] = float(alg.CalculateRMSD(mdl_rna, ref_rna))

if reference_interface.atom_count == 0:
    output["irmsd_rna_fit"] = None
else:
    output["irmsd_rna_fit"] = float(alg.CalculateRMSD(mdl_view_interface, ref_view_interface))

if reference_interface.atom_count == 0:
    output["irmsd_interface_fit"] = None
else:
    alg.SuperposeSVD(mdl_view_interface, ref_view_interface, apply_transform=True)
    output["irmsd_interface_fit"] = float(alg.CalculateRMSD(mdl_view_interface, ref_view_interface))

print("#"*15)

##################

pprint(output)

print("#"*15)

with open(output_json, "w") as f: json.dump(output, f, indent=4)

print('written!')

print("#"*15)

##################

