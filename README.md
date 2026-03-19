# Inferring the qualities of protein-RNA models with graph transformers

## Abstract
[Abstract goes here]

## Installation
[Installation goes here]

## Basic Usage

Running CARP requires specific formatting for the input files:

#### **`target_src/`** (Sequence-Derived Features)
This directory contains the target-level features and reference files.
* Must contain `rna.fasta` and `prot.fasta`.
* Must include monomeric protein reference `.pdb` files. We used relaxed AlphaFold2 predictions via [ColabFold](https://github.com/sokrypton/colabfold). The files must match the fasta IDs in the `prot.fasta` file.

#### **`model_src/`** (Structure-Derived Features)
This directory contains the model-level features.
* Must contain `model.pdb`.

### **Example Input** 
**`prot.fasta`**
```
>p0
PQYQTWEEFSRAAEKLYLADPMKARVVLKYRHSDGNLCVKVTDDLVCLVYKTDQAQDVKKIEKFHSQLMRLMVAKEARNVTMETE
>p1
VLLESEQFLTELTRLFQKCRTSGSVYITLKKYDGRTKPIPKKGTVEGFEPADNKCLLRATDGKKKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKTKKTK
```
```
├── {target_src}/
│   ├── rna.fasta
│   ├── prot.fasta
│   ├── p0.pdb
│   └── p1.pdb
└── {model_src}/
    └── model.pdb
```

1. **Generate Features**  

```
python run_tools.py -target_src {target_src} -model_src {model_src}
```

2. **Quality Score Inferenece**   

```
python run.py -target_src {target_src} -model_src {model_src}
```

  The CARP predicted qualities can be found @:
  > *`{model_src}/predicted_quality/carp.csv`* and *`{model_src}/predicted_quality/carp.pkl`

### **Example Output** 

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


