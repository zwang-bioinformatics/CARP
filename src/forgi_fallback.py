# see https://github.com/ViennaRNA/forgi/blob/master/forgi/threedee/model/coarse_grain.py#L547
# see https://github.com/ViennaRNA/forgi/blob/master/forgi/threedee/utilities/pdb.py#L628

import sys
import forgi.threedee.utilities.pdb as ftup
import forgi.graph.residue as fgr

fl = sys.argv[1]
out_dir = sys.argv[2]

with open(out_dir + "/fallback.cg",'w') as out_cg:
    chains, mr, interacting_residues = ftup.get_all_chains(fl)
    forgi_resids = [fgr.resid_from_biopython(r) for r in interacting_residues]
    lns = []
    for res in sorted(forgi_resids):
        chain_id = res.chain
        res_num = res.resid[1] 
        # print(f"interacting {chain_id}:{res_num}")
        lns += [f"interacting {chain_id}:{res_num}"]
    out_cg.write("\n".join(lns))