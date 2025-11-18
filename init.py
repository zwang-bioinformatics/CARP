ROOT = "/home/asiciliano/CARP/"

IPKNOT_PATH = f"{ROOT}tools/ipknot/build/ipknot"

AMIGOS_PATH = f"{ROOT}tools/"

LINEAR_PARTITION_PATH = f"{ROOT}tools/"

RNAVIEW_PATH = f"{ROOT}tools/RNAView/"

NSP3_PATH = f"{ROOT}tools/"

NSP_ENV = "nsp3"

nf_keys = [
    
    "sequence",
    "polymer_kind",
    "prot_rna_interface",
    
    "discounted_dist_chain_com",
    "unscaled_dist_chain_com", 
    "unscaled_dist_model_com", 
    "scaled_dist_model_com",
    "scaled_dist_chain_com",

    "angle_chain_com",
    "angle_model_com",

    "amigos_eta",
    "amigos_theta",

    "interacting_forgi",
    "interacting_rnaview",
    "interacting_ipknot",

    "psi_model_dssp",
    "phi_model_dssp",
    "rsa_model_dssp",
    "secondary_structure_model_dssp",
    
    "psi_netsurfp",
    "phi_netsurfp", 
    "rsa_netsurfp", 
    "p_q3_C_netsurfp",
    "p_q3_E_netsurfp",
    "p_q3_H_netsurfp",
    "disorder_netsurfp",
    
]
ef_keys = [
    
    "model_atom_distance",
    "model_bb_distance",
    "model_angle",

    "ref_distance",            
    "ref_angle",

    "linear_partition_prob", 
    "ipknot_pair", 
    "rnaview_pair",

    "is_inter_chain",
    "is_prot_rna",

]

ANGLE_FEATS = set([
    
    #### Node Angles ####
    
    "angle_prot_com",
    "angle_chain_com",

    "amigos_eta",
    "amigos_theta",

    "psi_model_dssp",
    "phi_model_dssp",
    
    "psi_netsurfp",
    "phi_netsurfp",

    #### Edge Angles ####

    "model_angle", 
    "ref_angle", 
    "angle_model_com"

])
