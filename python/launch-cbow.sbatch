#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -t24:00:00

# Make sure we have access to HPC-managed libraries.
module load tensorflow/python2.7/20170218

# Run.
PYTHONPATH=$PYTHONPATH:. python cbow/cbow.py cbow

!/bin/bash
