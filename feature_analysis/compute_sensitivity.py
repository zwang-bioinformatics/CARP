###################################

import os

os.environ['LD_LIBRARY_PATH'] = "/home/asiciliano/anaconda3/envs/analysis/lib"
os.environ['NUMEXPR_NUM_THREADS'] = '64'
import numexpr as ne 

import scienceplots

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

###################################

feat_map = {
    frozenset({'PROT', 'sequence'}): ["Amino Acid Sequence", "Source: Sequence\nPolymer: Protein"],
    frozenset({'scaled_dist_model_com', 'unscaled_dist_chain_com', 'PROT'}): ["Center of Mass Distances (Protein)", "Source: Model\nPolymer: Protein"],
    frozenset({'PROT', 'angle_model_com', 'angle_chain_com'}): ["Center of Mass Angles (Protein)", "Source: Model\nPolymer: Protein"],
    frozenset({'rsa_model_dssp', 'PROT'}): ["Annotated Relative Solvent Accessibility", "Source: Model\nPolymer: Protein"],
    frozenset({'PROT', 'secondary_structure_model_dssp', 'phi_model_dssp', 'psi_model_dssp'}): [r"Annotated Q8 (SS) and Torsion Angles ($\phi$, $\psi$)", "Source: Model\nPolymer: Protein"],
    
    frozenset({'PROT', 'rsa_netsurfp'}): ["Predicted Relative Solvent Accessibility", "Source: Sequence\nPolymer: Protein"],
    frozenset({'PROT', 'disorder_netsurfp'}): ["Predicted Disorder", "Source: Sequence\nPolymer: Protein"],
    frozenset({'phi_netsurfp', 'psi_netsurfp', 'p_q3_C_netsurfp', 'p_q3_H_netsurfp', 'PROT', 'p_q3_E_netsurfp'}): [r"Predicted Q3 (SS) and Torsion Angles ($\phi$, $\psi$)", "Source: Sequence\nPolymer: Protein"],

    frozenset({'sequence', 'RNA'}): ["Nucleic Acid Sequence", "Source: Sequence\nPolymer: RNA"],
    frozenset({'scaled_dist_model_com', 'unscaled_dist_chain_com', 'RNA'}): ["Center of Mass Distances (RNA)", "Source: Model\nPolymer: RNA"],
    frozenset({'RNA', 'angle_model_com', 'angle_chain_com'}): ["Center of Mass Angles (RNA)", "Source: Model\nPolymer: RNA"],
    frozenset({'amigos_eta', 'amigos_theta', 'RNA'}): [r"Annotated RNA Torsion Angles ($\eta$, $\theta$)", "Source: Model\nPolymer: RNA"],

    frozenset({'RNA', 'interacting_rnaview', 'interacting_forgi'}): ["Annotated Interacting Nucleotides", "Source: Model\nPolymer: RNA"],
    frozenset({'interacting_ipknot', 'RNA'}): ["Predicted Interacting Nucleotide", "Source: Sequence\nPolymer: RNA"],

    frozenset({'PROT-PROT', 'ref_distance'}): ["Predicted Intra-Protein Distance", "Source: Sequence\nPolymer: Protein"],
    frozenset({'ref_angle', 'PROT-PROT'}): ["Predicted Intra-Protein Angle", "Source: Sequence\nPolymer: Protein"],

    frozenset({'model_atom_distance', 'PROT-PROT', 'model_bb_distance'}): ["Model Distances (P-P)", "Source: Model\nPolymer: Protein"],
    frozenset({'RNA-RNA', 'model_atom_distance', 'model_bb_distance'}): ["Model Distances (R-R)", "Source: Model\nPolymer: RNA"],
    frozenset({'PROT-RNA', 'model_atom_distance', 'model_bb_distance'}): ["Model Distances (R-P)", "Source: Model\nPolymer: Both"],

    frozenset({'PROT-PROT', 'model_angle'}): ["Model Angle (P-P)", "Source: Model\nPolymer: Protein"],
    frozenset({'RNA-RNA', 'model_angle'}): ["Model Angle (R-R)", "Source: Model\nPolymer: RNA"],
    frozenset({'PROT-RNA', 'model_angle'}): ["Model Angle (R-P)", "Source: Model\nPolymer: Both"],

    frozenset({'RNA-RNA', 'rnaview_pair'}): ["Annotated Base-Pair", "Source: Model\nPolymer: RNA"],
    frozenset({'RNA-RNA', 'linear_partition_prob'}): ["Predicted Base-Pair Probability", "Source: Sequence\nPolymer: RNA"],
    frozenset({'ipknot_pair', 'RNA-RNA'}): ["Predicted Base-Pairing", "Source: Sequence\nPolymer: RNA"],

}

fcolors = {
    "Source: Model\nPolymer: Protein": "#0345fc",
    "Source: Sequence\nPolymer: Protein": "#03befc",
    "Source: Model\nPolymer: RNA": "#f55142",
    "Source: Sequence\nPolymer: RNA": "#fcf403",
    "Source: Model\nPolymer: Both": "#34eb6e"
}

###################################

importance = {
    "target": [],
    "CARP": [],
    "delta": [],
    "abs_delta": [],
    "src_dir": [],
    "features": [],
    "features_category": [],
    "polymer_kind": [],
    "size": [],
    "component_type": []
}

###################################

dataset = "DOCKING"

for target in os.listdir(f"/home/asiciliano/CARP/data/targets/{dataset}/"):

    target_src = f"/home/asiciliano/CARP/data/targets/{dataset}/{target}/"
    
    if not os.path.exists(target_src + "/carp_abilation.pkl"): 
        continue

    print("Loading Target:", target)

    ###################################

    data = pd.read_pickle(target_src + "/carp_abilation.pkl")
    data = data[data["config"].str.contains("fold")]
    data["features"] = data["features"].apply(lambda f: frozenset(f))
    # print(data["features"].unique())

    ###################################

    data["Iface"] = (data["ics"] + data["ips"] + data["ilddt"] ) / 3
    data["Fold"] = (data["bb_lddt"] + data["oligo_gdtts"] + data["oligo_gdtha"] ) / 3
    data["Merged"] = (data["Fold"] + data["Iface"]) / 2
    data["RP"] = data.apply(
        lambda r: None
            if not len(r["interface"]) else (
                (r["interface_ics_pred"] + r["interface_ips_pred"])/2
            ).mean(), 
        axis=1
    )

    # print(data[data["sample"] == "base"])

    data = data.groupby(["sample", "features", "src_dir"]).mean(numeric_only=True).reset_index()
    
    dg = data.groupby(["features", "src_dir"])

    for group, df in tqdm(dg, total = len(dg)):
        feat_tag, feat_cat = feat_map[group[0]]#" & ".join(sorted(group[0]))
        # print(group[0])
        # print(feat_tag, feat_cat)

        base = df[df["sample"] == "base"]
        # print(len(base))

        # print(df["sample"].unique())
        assert len(base) == 1
        base = base.iloc[0].to_dict()
        df = df[df["sample"] != "base"]
        
        for score_type in ["Merged", "Fold", "Iface", "RP"]: # RP
            importance["target"] += [target]*len(df)
            importance["features"] += [feat_tag]*len(df)
            importance["features_category"] += [feat_cat]*len(df)

            importance["src_dir"] += [group[1]]*len(df)
            importance["CARP"] += [score_type]*len(df)
            importance["delta"] += (base[score_type] - df[score_type]).to_list()
            importance["abs_delta"] += (base[score_type] - df[score_type]).abs().to_list()
            importance["polymer_kind"] += [['PROT', 'RNA'][int('RNA' in group[0])]]*len(df)
            importance["component_type"] += [['Node', 'Edge'][int(
                'RNA-RNA' in group[0] or 'PROT-RNA' in group[0] or 'PROT-PROT' in group[0]
            )]]*len(df)

            importance["size"] += df["size"].to_list()

importance = pd.DataFrame(importance)

importance['ad_quantile'] = importance.groupby(['CARP', 'src_dir', 'component_type'])['abs_delta'].rank(method = "max", pct=True)

###################################

sns.set_context("paper") 
sns.set_style("white")

with plt.style.context('nature'): 

    g = sns.catplot(
        data=importance,

        y='features',
        # kind='count',

        # x='abs_delta',
        x = 'ad_quantile',
        kind='boxen',
        # kind = 'violin',    
        hue='features_category', 
        hue_order=[
            "Source: Sequence\nPolymer: Protein",
            "Source: Model\nPolymer: Protein",
            "Source: Sequence\nPolymer: RNA",
            "Source: Model\nPolymer: RNA",
            "Source: Model\nPolymer: Both"
        ],
        col='CARP', col_wrap = 2,
        sharex=False, sharey=False,
        dodge=False,
        height=7, aspect=1.75,
        palette=fcolors,
    )

    g.set_xticklabels(fontsize=16)
    g.set_yticklabels(fontsize=16)
    g.set_titles("CARP-{col_name}", size = 22, pad=18)

    for ax in g.axes.flat:
        # ax.set_xscale('logit')
        ax.set_xlim(0,1)
        ax.tick_params(labelbottom=True, axis='x', labelsize=16, rotation=0)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        ax.set_xlabel(r"Prediction Sensitivity $Q\left(|\Delta|\right)$", fontsize=17.5, visible=True, labelpad=10)
        ax.set_ylabel('')
        
    # g.add_legend(title="Feature Category", title_fontsize=18, fontsize=15)
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(0.5, -0.085),
        ncol=5,
        columnspacing=3.5, 
        handletextpad=0.5, 
        handleheight=0.1,
        borderpad=0.1,
        title=None, 
        frameon=False,
        fontsize=20
    )
    type_of_struct = "Models" if dataset == "CASP16" else "Decoys"

    title_main = f"Permutation-based sensitivity analysis for {dataset.title() if dataset == "DOCKING" else dataset} {type_of_struct}"
    title_sub = r"Sensitivity is defined as the within-model quantile of the change in prediction under random feature-specific permutations"

    tx = 0.98
    g.figure.suptitle(f"{title_main}\n", fontsize=28, fontweight='black', x=tx, y=0.991, ha = 'right')
    g.figure.text(tx, 0.945, title_sub, ha='right', fontsize=20, fontstyle='italic', alpha=0.8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.95, hspace=0.3) 

    plt.savefig(
        f'./sensitivity_{dataset}.png',
        dpi=300, bbox_inches = 'tight', transparent = False
    )

###################################
