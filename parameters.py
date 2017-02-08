import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_encode", action="store_true", default=False, dest="use_encode")
#parser.add_argument("--use_reinforce", action="store_true", default=False, dest="use_reinforce")
#parser.add_argument("--rl_baseline", type=str, default="ema")
parser.add_argument("--runs", type=int, default=4)
parser.add_argument("--datapath", type=str, default="../data")
parser.add_argument("--logpath", type=str, default="../logs")
#parser.add_argument("--venv", type=str, default="~/spinn/.venv-hpc/bin/activate")
args = parser.parse_args()

def load_parameters():
    FIXED_PARAMETERS = {
        "data_type":     "snli",
        "training_data_path":    "../data/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
        "dev_data_path":    "../data/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
        "test_data_path":    "../data/snli_1.0/snli_1.0_test.jsonl".format(args.datapath),
        "embedding_data_path": "../data/glove.6B.50d.txt".format(args.datapath),
        "embeddings_to_load": 50000,
        "log_path": "{}".format(args.logpath),
        "word_embedding_dim":   50,
        "seq_length":   25,
        #"eval_seq_length":  "50",
        #"eval_interval_steps": "500",
        #"statistics_interval_steps": "500",
        #"use_internal_parser": "",
        "batch_size":  32,
        #"ckpt_path":  "{}".format(args.logpath)
    }

    return FIXED_PARAMETERS