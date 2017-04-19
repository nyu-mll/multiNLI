import argparse
import os
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import print_tensors_in_checkpoint_file

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="Give model name, example cbow, spinn. etc.")
parser.add_argument("--logpath", type=str, default="./logs", help="Give path to logs/checkpoints")

args = parser.parse_args()

path = os.path.join(args.logpath, args.model_name) + ".ckpt_best"
print_tensors_in_checkpoint_file(file_name=path, tensor_name='')