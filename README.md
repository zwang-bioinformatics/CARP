# Inferring the qualities of protein-RNA models with graph transformers

### **Abstract**

### **Installation**

### **Basic Usage**

Requirements for running CARP:

* **`target_src`**

  This is the path for the target-level (sequence-derived) features. 
    It must contain *`{target_src}/rna.fasta`* and *`{target_src}/prot.fasta`*.
    You must also provide monomeric protein reference *.pdb* files (ideally using colabfold),
    and name them as *`{target_src}/{FASTA-ID}.pdb`*, where *`{FASTA-ID}`* is the corresponding
    ID found for the respective sequence in the *`{target_src}/prot.fasta`*,
    e.g., *`{target_src}/p0.pdb`*, *`{target_src}/p1.pdb`*, ...

* **`model_src`**

  This is the path for the model level (structure-derived) feaures, and must contain *`{model_src}/model.pdb`*. 

1. **Generate Features**  

```
python run_tools.py -target_src {target_src} -model_src {model_src}
```

2. **Quality Score Inferenece**   

```
python run.py -target_src {target_src} -model_src {model_src}
```

  The CARP predicted qualities can be found @:
  > *`{model_src}/predicted_quality/carp.csv`* and *`{model_src}/predicted_quality/carp.pkl`*
