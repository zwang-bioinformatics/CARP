###################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p","--pdb", action="store", type=str, help="pdb file", required=True)
parser.add_argument("-o","--out_dir", action="store", type=str, help="output directory",required=True)
parser.add_argument("-b","--bin", action="store", type=str, help="bin",required=True)

args = parser.parse_args()

print("PDB:",args.pdb)
print("OUT:",args.out_dir)

###################################

import os
import json
import subprocess

###################################

RNAVIEW = args.bin #"/home/asiciliano/RNA-QA/src/resources/RNAView/bin/"

###################################

os.chdir(RNAVIEW)

command = [RNAVIEW + "rnaview","-f",args.pdb]

print(" ".join(command))

print(subprocess.getoutput(" ".join(command)))

for ext in [".xml",".ps","_new_torsion.out",".out","_tmp.pdb","_sort.out"]:

    if os.path.exists(args.pdb + ext): 
        command = ["mv", args.pdb + ext, args.out_dir]
        subprocess.getoutput(" ".join(command))
    