# Baseline Models for MultiNLI Corpus

This is the code we used to establish baselines for the MultiNLI corpus introduced in [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/pdf/1704.05426.pdf).

## Data
The MultiNLI and SNLI corpora are both distributed in JSON lines and tab separated value files. Both can be downloaded [here](https://www.nyu.edu/projects/bowman/multinli/).

## Models
We present three baseline neural network models. These range from a bare-bones model (CBOW), to an elaborate model which has achieved state-of-the-art performance on the SNLI corpus (ESIM),

- Continuous Bag of Words (CBOW):  in this model, each sentence is represented as the sum of the embedding representations of its
words. This representation is passed to a deep, 3-layers, MLP. Main code for this model is in [`cbow.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/models/cbow.py)
- Bi-directional LSTM: in this model, the average of the states of
a bidirectional LSTM RNN is used as the sentence representation. Main code for this model is in [`bilstm.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/models/bilstm.py)
- Enhanced Sequential Inference Model (ESIM): this is our implementation of the [Chen et al.'s (2017)](https://arxiv.org/pdf/1609.06038v2.pdf) ESIM, without ensembling with a TreeLSTM. Main code for this model is in [`esim.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/models/esim.py)

We use dropout for regularization in all three models.

## Training and Testing

### Training settings

The models can be  trained on three different settings. Each setting has its own training script.

- To train a model only on SNLI data, 
	- Use [`train_snli.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/train_snli.py). 
	- Accuracy on SNLI's dev-set is used to do early stopping. 

- To train a model on only MultiNLI or on a mixture of MultiNLI and SNLI data, 
	- Use [`train_mnli.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/train_mnli.py). 
	- The optional `alpha` flag determines what percentage of SNLI data is used in training. The default value for alpha is 0.0, which means the model will be only trained on MultiNLI data. 
	- If `alpha` is a set to a value greater than 0 (and less than 1), an `alpha` percentage of SNLI training data is randomly sampled at the beginning of each epoch. 
	- When using SNLI training data in this setting, we set `alpha` = 0.15.
	- Accuracy on MultiNLI's matched dev-set is used to do early stopping.

- To train a model on a single MultiNLI genre, 
	- Use [`train_genre.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/train_genre.py). 
	- To use this training setting, you must call the `genre` flag and set it to a valid training genre (`travel`, `fiction`, `slate`, `telephone`, `government`, or `snli`). 
	- Accuracy on the dev-set for the chosen genre is used to do early stopping. 
	- Additionally, logs created with this training setting contain evaulation statistics by genre. 
	- You can also train a model on SNLI with this script if you desire genre specific statistics in your logs. 

### Command line flags

To start training with any of the training scripts, there are a couple of required command-line flags and an array of optional flags. The code concerning all flags can be found in [`parameters.py`](https://github.com/NYU-MLL/multiNLI/blob/master/python/util/parameters.py). All the parameters set in `parameters.py` are printed to the log file everytime the training script is launched. 

Required flags,

- `model_type`: there are three model types in this repository, `cbow`, `bilstm`, and `cbow`. You must state which model you want to use.
- `model_name`: this is your experiment name. This name will be used the prefix the log and checkpoint files. 

Optional flags,

- `datapath`: path to your directory with MultiNLI, and SNLI data. Default is set to "../data"
- `ckptpath`: path to your directory where you wish to store checkpoint files. Default is set to "../logs"
- `logpath`: path to your directory where you wish to store log files. Default is set to "../logs"
- `emb_to_load`: path to your directory with GloVe data. Default is set to "../data"
- `learning_rate`: the learning rate you wish to use during training. Default value is set to 0.0004
- `keep_rate`: the hyper-parameter for dropout-rate. `keep_rate` = 1 - dropout-rate. The default value is set to 0.5.
- `seq_length`: the maximum sequence length you wish to use. Default value is set to 50. Sentences shorter than `seq_length` are padded to the right. Sentences longer than `seq-length` are truncated. 
- `emb_train`: boolean flag that determines if the model updates word embeddings during training. If called, the word embeddings are updated. 
- `alpha`: only used during `train_mnli` scheme. Determines what percentage of SNLI training data to use in each epoch of training. Default value set to 0.0 (which makes the model train on MultiNLI only).
- `genre`: only used during `train_genre` scheme. Use this flag to set which single genre you wish to train on. Valid genres are `travel`, `fiction`, `slate`, `telephone`, `government`, or `snli`.
- `test`: boolean used to test a trained model. Call this flag if you wish to load a trained model and test it on MultiNLI dev-sets* and SNLI test-set. When called, the best checkpoint will be used (see section on checkpoints for more details).

 
*Dev-sets are currently used for testing on MultiNLI since the test-sets have not be released. 

### Other parameters

Remaining parameters like the size of hidden layers, word embeddings, and minibatch can be changed directly in `parameters.py`. The default hidden embedding and word embedding size is set to 300, the minibatch size (`batch_size` in the code) is set to 32.

### Sample commands
To execute all of the following sample commands, you must be in the "python" folder,

- To train on SNLI data only, here is a sample command,

	`PYTHONPATH=$PYTHONPATH:. python train_snli.py cbow petModel-0 --keep_rate 0.9 --seq_length 25 --emb_train`

	where the `model_type` flag is set to `cbow` and can be swapped for `bilstm` or `esim`, and the `model_name` flag is set to `petModel-0` and can be changed to whatever you please.

- Similarly, to train on a mixture MultiNLI and SNLI data, here is a sample command,

	`PYTHONPATH=$PYTHONPATH:. python train_mnli.py bilstm petModel-1 --keep_rate 0.9 --alpha 0.15 --emb_train`

	where 15% of SNLI training data is randomly sampled at the beginning of each epoch. 

- To train on just the `travel` genre in MultiNLI data,

	`PYTHONPATH=$PYTHONPATH:. python train_genre.py esim petModel-2 --genre travel --emb_train`

### Testing models

#### On dev set,
To test a trained model, simply add the `test` flag to the command used for training. The best checkpoint will be loaded and used to evaluate the model's performance on the MultiNLI dev-sets, SNLI test-set, and the dev-set for each genre in MultiNLI.

For example,

`PYTHONPATH=$PYTHONPATH:. python train_genre.py esim petModel-2 --genre travel --emb_train --test`


With the `test` flag, the `train_mnli.py` script will also generate a CSV of predictions for the unlabaled matched and mismatched test-sets.

#### Results for unlabeled test sets,
To get a CSV of predicted results for unlabeled test sets use `predictions.py`. This script requires the same flags as the training scripts. You must enter the `model_type` and `model_name`, and the path to the saved checkpoint and log files if they are different from the default (the default is set ot `../logs` for both paths). 

Here is a sample command,

`PYTHONPATH=$PYTHONPATH:. python predictions.py esim petModel-1 --alpha 0.15 --emb_train --logpath ../logs_keep --ckptpath ../logs_keep `

This script will create a CSV with two columns: pairID and gold_label.


### Checkpoints 

We maintain two checkpoints: the most recent checkpoint and the best checkpoint. Every 500 steps, the most recent checkpoint is updated, and we test to see if the dev-set accuracy has improved by at least 0.04%. If the accuracy has gone up by at least 0.04%, then the best checkpoint is updated.

## License

Copyright 2017, New York University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

