# Inferring the qualities of protein-RNA models with graph transformers
<p align="center"><img src=".github/pipeline.png"/></p>

Andrew Jordan Siciliano, Yifan Bao, Bishal Shrestha, Zheng Wang, Inferring the qualities of protein-RNA models with graph transformers, Bioinformatics, 2026;, btag202, https://doi.org/10.1093/bioinformatics/btag202

## Abstract
**Motivation:** Breakthrough advancements in protein tertiary and quaternary structure prediction have
accelerated structural bioinformatics research activity and drug development processes. However, many
biological mechanisms involve more complicated interactions, such as those between amino and nucleic acids.
Predicting the structure of protein-RNA complexes is highly relevant and challenging due to data scarcity
and experimental difficulties. Understanding and interpreting these interactions can yield crucial insights
into various human diseases and biological phenomena. Thus, quality assessment methods that specifically
evaluate protein-RNA complex models can provide significant utility in this emerging area of protein-RNA
structural bioinformatics research.

**Results:** We propose a novel graph transformer-based approach named CARP (complex quality assessment
of RNA and protein) to infer multiple quality perspectives of protein-RNA complex models. For a single
protein-RNA complex model, in one shot, CARP simultaneously predicts multiple overall fold, overall interface,
and per-protein-RNA interface quality estimates. When evaluated against a non-redundant protein-RNA
docking benchmark, our methods demonstrated obvious improved performance compared to almost all of
the existing scoring tools, particularly when ordering and selecting the highest quality decoys. Furthermore,
CARP consistently selected higher quality models relative to other predictors when tested on CASP16 targets.
Specifically, CARP-predicted global interface and global protein-RNA interface qualities were ranked first
and second, respectively, based on the selected top-3 models over all ten CASP16 protein-RNA complex
targets. CARP also showed a strong ability, compared to both existing tools and AlphaFold3 self-estimates, in
selecting high quality AlphaFold3 models.

## Installation

#### Step 1:
CARP Conda Environment:
```
conda create -n CARP python=3.9
conda activate CARP
conda install salilab::dssp
conda install -c conda-forge boost-cpp=1.73.0
pip install biopython==1.79
pip install logging-exceptions==0.1.9
pip install --only-binary :all: "appdirs>=1.4" cython "future" "pandas>=0.20" "scipy>=0.19.1" numpy==1.26.4 numba more_itertools
pip install forgi==2.2.3 --only-binary :all: --no-deps
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.5.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu118.html
```
#### Step 2:
Clone the CARP repo:
```
git clone https://github.com/zwang-bioinformatics/CARP.git
cd ./CARP
```
> [!IMPORTANT]
Note `ROOT` in `init.py` should point to the absolute path of `./CARP`.

#### Step 3:

Install the External Tools:
- Install [NetSurfP-3.0](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/), then update the `NSP_ENV` & `NSP3_PATH` in `init.py` accordingly. 

- Install [IPKnot](https://github.com/satoken/ipknot), then update the `IPKNOT_PATH` in `init.py` accordingly. 

- Install [AMIGOS](https://github.com/pylelab/AMIGOS/), then update the `AMIGOS_PATH` in `init.py` accordingly. 

- Install [LinearPartition](https://github.com/LinearFold/LinearPartition/), then update the `LINEAR_PARTITION_PATH` in `init.py` accordingly. 

- Install [RNAView](https://github.com/rcsb/RNAView/), then update the `RNAVIEW_PATH` in `init.py` accordingly. 

- Install [MCAnnotate](https://major.iric.ca/MajorLabEn/MC-Tools.html):

  Download and unzip [MC-Annotate.zip](https://major.iric.ca/MajorLabEn/MC-Tools_files/MC-Annotate.zip).
  Put the MC-Annotate executable in `./tools/`.

  To use MC-Annotate in your current session:
  ```
  export PATH="$PATH:./tools"
  ```
  Alternatively, you can permanently add MC-Annotate your path:
  ```
  echo 'export PATH="$PATH:./tools"' >> ~/.bashrc
  source ~/.bashrc
  ```

> [!IMPORTANT]
Update the paths in `init.py` accordingly.

## Usage

> [!NOTE]
Running CARP requires specific formatting for the input files

##### `target_src/` (sequence-derived features)
This directory points to the location for target-level features and reference file/s.
* Must contain `rna.fasta` and `prot.fasta`.
* Must contain monomeric protein reference `.pdb` files. We recommend using relaxed AlphaFold2 prediction/s via [ColabFold](https://github.com/sokrypton/colabfold). The file/s must match the fasta ID/s in the `prot.fasta` file (for reproducibility we have provided these for the blind-test complexes).

##### `model_src/` (structure-derived features)
This directory points to the location for model-level features.
* Must contain `model.pdb`.

#### Generate Features:

```
python run_tools.py -target_src {target_src} -model_src {model_src}
```

#### Perform Quality Score Inference:

```
python run.py -target_src {target_src} -model_src {model_src}
```

  The CARP predicted qualities can be found @:
  > *`{model_src}/predicted_quality/carp.csv`* and *`{model_src}/predicted_quality/carp.pkl`*

#### **Example**:

This is an example case for the CASP16 target `M1209` and model `M1209TS006_1`.

Fasta `./data/example/prot.fasta`
```
>p0
EISEVQLVESGGGLVQPGGSLRLSCAASGFYISYSSIHWVRQAPGKGLEWVASISPYSGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARQGYRRRSGRGFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHT
>p1
SDIQMTQSPSSLSASVGDRVTITCRASQSVSSAVAWYQQKPGKAPKLLIYSASSLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQSYSFPSTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSADSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC
```
Fasta `./data/example/rna.fasta`
```
>r0
GGGCCGGGCGCGGUGGCGCGCGCCUGUAGUCCCAGCUACUCGGGAGGCUC
```

Inputs:
```
├── ./data/example/ (target_src)
│   ├── rna.fasta
│   ├── prot.fasta
│   ├── p0.pdb (AlphaFold2 reference prediction for sequence p0)
│   └── p1.pdb (AlphaFold2 reference prediction for sequence p1)
└── ./data/example/example_model/ (model_src)
    └── model.pdb
```

Commands:
```
python run_tools.py -target_src ABSOLUTE_PATH/CARP/data/example/ -model_src ABSOLUTE_PATH/CARP/data/example/example_model/
python run.py -target_src ABSOLUTE_PATH/CARP/data/example/ -model_src ABSOLUTE_PATH/CARP/data/example/example_model/
```

Outputs:
 ```
├── ./data/example/ (target_src)
│   ├── rna.fasta
│   ├── prot.fasta
│   ├── p0.pdb
│   ├── p1.pdb
│   ├── bp.mat
│   ├── out.bpseq
│   └── nsp/
│       └── 01/
│           └── 01.csv
├── ./data/example/example_model/ (model_src)
│   ├── model.pdb
│   ├── dssp.npy
│   ├── agged_features.npy
│   ├── RNAView_out/
│   │   └── ...
│   ├── forgi_out/
│   │   └── ...
│   ├── amigos_output/
│   │    └── ...
│   └── predicted_quality/
│       └── carp.csv
│       └── carp.pkl
```

The generated output can be compared with the expected outputs,

`./data/example/expected_model_output.csv` and `./data/example/expected_model_output.csv`,

to confirm everything is functional.

## Data (coming soon)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.19227794.svg)](https://doi.org/10.5281/zenodo.19227794)

## **Citation**  

```
@article{siciliano2026inferring,
  title={Inferring the qualities of protein-RNA models with graph transformers},
  author={Siciliano, Andrew Jordan and Bao, Yifan and Shrestha, Bishal and Wang, Zheng},
  journal={Bioinformatics},
  pages={btag202},
  year={2026},
  publisher={Oxford University Press}
}
```


