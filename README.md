# Inferring the qualities of protein-RNA models with graph transformers
<p align="center"><img src=".github/pipeline.png"/></p>

## Abstract
[Abstract goes here]

## Installation
[Installation goes here]

## Usage

> [!NOTE]
Running CARP requires specific formatting for the input files

##### `target_src/` (sequence-derived features)
This directory points to the location for target-level features and reference file/s.
* Must contain `rna.fasta` and `prot.fasta`.
* Must include monomeric protein reference `.pdb` files. We used relaxed AlphaFold2 predictions via [ColabFold](https://github.com/sokrypton/colabfold). The files must match the fasta IDs in the `prot.fasta` file.

##### `model_src/` (structure-derived features)
This directory points to the location for model-level features.
* Must contain `model.pdb`.

### Generate Features 

```
python run_tools.py -target_src {target_src} -model_src {model_src}
```

### Perform Quality Score Inference 

```
python run.py -target_src {target_src} -model_src {model_src}
```

  The CARP predicted qualities can be found @:
  > *`{model_src}/predicted_quality/carp.csv`* and *`{model_src}/predicted_quality/carp.pkl`*

### **Example** 

Fasta `prot.fasta`
```
>p0
PQYQTWEEFSRAAEKLYLADPMKARVVLKYRHSDGNLCVKVTDDLVCLVYKTDQAQDVKKIEKFHSQLMRLMVAKEARNVTMETE
>p1
VLLESEQFLTELTRLFQKCRTSGSVYITLKKYDGRTKPIPKKGTVEGFEPADNKCLLRATDGKKKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKTKKTK
```
Fasta `rna.fasta`
```
>r0
GGGCCGGGCGCGGUGGCGCGCGCCUGUAGUCCCAGCUACUCGGGAGGCUC
```

Inputs:
```
├── {target_src}/
│   ├── rna.fasta
│   ├── prot.fasta
│   ├── p0.pdb (AlphaFold2 reference prediction for sequence p0)
│   └── p1.pdb (AlphaFold2 reference prediction for sequence p1)
└── {model_src}/
    └── model.pdb
```

Outputs:
 ```
├── {target_src}/
│   ├── rna.fasta
│   ├── prot.fasta
│   ├── p0.pdb
│   ├── p1.pdb
│   ├── bp.mat
│   ├── out.bpseq
│   ├── feats.log
│   └── nsp/
│       └── 01/
│           └── 01.csv
├── {model_src}/
│   ├── model.pdb
│   ├── dssp.npy
│   ├── agged_features.npy
│   ├── feats.log
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
### **Citation**  


[![DOI]()]()

