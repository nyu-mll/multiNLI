import argparse

parser = argparse.ArgumentParser()

models = ['ebim','cbow', 'bilstm']
def types(s):
    options = [mod for mod in models if s in models]
    if len(options) == 1:
        return options[0]
    return s

model_sub = ['ebim', 'ebim_noAvgPool', 'ebim_noDiffMul', 'ebim_noInfBiLSTM', 'bilstm', 'bilstm_meanpool', 'cbow_2layer', 'cbow']
def subtypes(s):
    options = [mod for mod in model_sub if s in model_sub]
    if len(options) == 1:
        return options[0]
    return s

genres = ['travel', 'fiction', 'slate', 'telephone', 'government']
def subtypes(s):
    options = [mod for mod in genres if s in genres]
    if len(options) == 1:
        return options[0]
    return s

parser.add_argument("model_type", choices=models, type=types, help="Give model type.")
parser.add_argument("model_subtype", choices=model_sub, type=subtypes, help="Give model subtype")
parser.add_argument("model_name", type=str, help="Give model name, this will name logs and checkpoints made. For example cbow, ebim2 etc.")

parser.add_argument("--data_type", type=str, default="snli", help="Give dataset name, example snli, multiNLI etc.")

parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--ckptpath", type=str, default="../logs")
parser.add_argument("--logpath", type=str, default="../logs")

parser.add_argument("--emb_to_load", type=int, default=None, help="Number of embeddings to load")
parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--keep_rate", type=float, default=0.5, help="Keep rate for dropout in the model")
parser.add_argument("--alpha", type=float, default=0., help="What percentage of SNLI data to use in training")
parser.add_argument("--genre", type=str, help="Which genre to train on")

parser.add_argument("--emb_train", action='store_true', help="Call if you want to make your word embeddings trainable.")
parser.add_argument("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")

args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "data_type": "{}".format(args.data_type),
        "model_type": args.model_type,
        "model_subtype": args.model_subtype,
        "model_name": args.model_name,
        "training_mnli": "{}/multinli_0.9/multinli_0.9_train.jsonl".format(args.datapath),
        "dev_matched": "{}/multinli_0.9/multinli_0.9_dev_matched.jsonl".format(args.datapath),
        "dev_mismatched": "{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl".format(args.datapath),
        "test_matched": "{}/multinli_0.9/multinli_0.9_test_matched.jsonl".format(args.datapath),
        "test_mismatched": "{}/multinli_0.9/multinli_0.9_test_mismatched.jsonl".format(args.datapath),
        "training_snli": "{}/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
        "dev_snli": "{}/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
        "test_snli": "{}/snli_1.0/snli_1.0_test.jsonl".format(args.datapath),
        "embedding_data_path": "{}/glove.840B.300d.txt".format(args.datapath),
        "log_path": "{}".format(args.logpath),
        "ckpt_path":  "{}".format(args.ckptpath),
        "embeddings_to_load": args.emb_to_load,
        "word_embedding_dim": 300,
        "hidden_embedding_dim": 300,
        "seq_length": 50,
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

