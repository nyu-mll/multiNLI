#!/bin/bash

#SBATCH --mem=12000
#SBATCH -t24:00:00

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 

# Run.
PYTHONPATH=. python ebim/ebim.py ebim
