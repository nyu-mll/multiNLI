#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -t24:00:00

# Make sure we have access to HPC-managed libraries.
module load tensorflow/python2.7/20170218

# Run.
PYTHONPATH=$PYTHONPATH:. python training_script model_type experiment_name --keep_rate 0.5 --learning_rate 0.0004 --alpha 0.13 --emb_train

# Available training_scripts: train_snli.py, train_mnli.py, train_genre.py
# Available model_types: cbow, bilstm, esim