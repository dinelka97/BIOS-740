#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5g
#SBATCH -t 1-

module purge
module add python/3.12.1

python code/snp.py

