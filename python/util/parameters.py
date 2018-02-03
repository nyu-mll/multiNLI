"""
The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.

Required arguements are,
model_type: which model you wish to train with. Valid model types: cbow, bilstm, and esim.
model_name: the name assigned to the model being trained, this will prefix the name of the logs and checkpoint files.
"""

import argparse
import io
import os
import json

parser = argparse.ArgumentParser()

models = ['esim','cbow', 'bilstm', 'lstm']
def types(s):
    options = [mod for mod in models if s in models]
    if len(options) == 1:
        return options[0]
    return s

# Valid genres to train on. 
genres = ['travel', 'fiction', 'slate', 'telephone', 'government']
def subtypes(s):
    options = [mod for mod in genres if s in genres]
    if len(options) == 1:
        return options[0]
    return s

parser.add_argument("model_type", choices=models, type=types, help="Give model type.")
parser.add_argument("model_name", type=str, help="Give model name, this will name logs and checkpoints made. For example cbow, esim_test etc.")

parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--ckptpath", type=str, default="../logs")
parser.add_argument("--logpath", type=str, default="../logs")

parser.add_argument("--emb_to_load", type=int, default=None, help="Number of embeddings to load. If None, all embeddings are loaded.")
parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--keep_rate", type=float, default=0.5, help="Keep rate for dropout in the model")
parser.add_argument("--seq_length", type=int, default=50, help="Max sequence length")
parser.add_argument("--emb_train", action='store_true', help="Call if you want to make your word embeddings trainable.")

parser.add_argument("--genre", type=str, help="Which genre to train on")
parser.add_argument("--alpha", type=float, default=0., help="What percentage of SNLI data to use in training")

parser.add_argument("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")

args = parser.parse_args()

"""
# Check if test sets are available. If not, create an empty file.
test_matched = "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl".format(args.datapath)

if os.path.isfile(test_matched):
    test_matched = "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl".format(args.datapath)
    test_mismatched = "{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl".format(args.datapath)
    test_path = "{}/multinli_0.9/".format(args.datapath)
else:
    test_path = "{}/multinli_0.9/".format(args.datapath)
    temp_file = os.path.join(test_path, "temp.jsonl")
    io.open(temp_file, "wb")
    test_matched = temp_file
    test_mismatched = temp_file
"""
# Check if test sets are available. If not, create an empty file.
test_matched = "{}/multinli_0.9/multinli_0.9_test_matched.jsonl".format(args.datapath)

if os.path.isfile(test_matched):
    test_matched = "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath) #"{}/multinli_0.9/multinli_0.9_test_matched.jsonl".format(args.datapath)
    test_mismatched = "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath) #"{}/multinli_0.9/multinli_0.9_test_mismatched.jsonl".format(args.datapath)
    test_path = "{}".format(args.datapath)
else:
    test_path = "{}".format(args.datapath)
    temp_file = os.path.join(test_path, "temp.jsonl")
    io.open(temp_file, "wb")
    test_matched = temp_file
    test_mismatched = temp_file


def load_parameters():
    FIXED_PARAMETERS = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl".format(args.datapath),
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath),
        "test_matched": test_matched,
        "test_mismatched": test_mismatched,
        "training_snli": "{}/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
        "dev_snli": "{}/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
        "test_snli": "{}/snli_1.0/snli_1.0_test.jsonl".format(args.datapath),
        "embedding_data_path": "{}/glove.840B.300d.txt".format(args.datapath),
        #"embedding_data_path": "{}/glove.6B.50d.txt".format(args.datapath),
        "log_path": "{}".format(args.logpath),
        "ckpt_path":  "{}".format(args.ckptpath),
        "embeddings_to_load": args.emb_to_load,
        "word_embedding_dim": 300,
        "hidden_embedding_dim": 300,
        #"word_embedding_dim": 50,
        #"hidden_embedding_dim": 50,
        "seq_length": args.seq_length,
        "keep_rate": args.keep_rate, 
        "batch_size": 32,
        "learning_rate": args.learning_rate,
        "emb_train": args.emb_train,
        "alpha": args.alpha,
        "genre": args.genre
    }

    return FIXED_PARAMETERS

def train_or_test():
    return args.test

