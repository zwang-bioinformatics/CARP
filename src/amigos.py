###################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-out_dir","--out_dir", action="store", type=str, help="dir", required=True)
parser.add_argument("-p","--perl", action="store", type=str, help="dir", required=True)

args = parser.parse_args()

print("OUT:",args.out_dir)

###################################

import os
import json
import subprocess

###################################

AMIGOS = args.perl #"/home/asiciliano/RNA-QA/src/resources/AMIGOS/AMIGOS.pl"

# echo /home/asiciliano/RNA-QA/vault/models/7Y/7YR6/CASP15/R1190TS035_1o/ | ./AMIGOS.pl
###################################

for fl in os.listdir(args.out_dir): 
    if fl in ["all_area.txt", "all_sprd.txt", "in.pdb_area.txt", "in.pdb_sprd.txt"]: os.remove(args.out_dir + fl)

os.chdir(args.out_dir)

command = [
    "echo",args.out_dir,
    "|",AMIGOS,
    ">", args.out_dir + "log.txt"
]

print(" ".join(command),"\n\n")

print(subprocess.getoutput(" ".join(command)))