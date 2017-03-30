#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -t24:00:00

# Make sure we have access to HPC-managed libraries.
module load tensorflow/python2.7/20170218

# Run.
PYTHONPATH=$PYTHONPATH:. python training_script model_type model_subtype experiment_name --keep_rate 0.5 --learning_rate 0.0004 --emb_train

# Example training_scripts: train.py, train_mnli.py
# Example model_types: cbow, ebim, bilstm
# Example model_subtypes: cbow_2layer, ebim, bilstm-attn