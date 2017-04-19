# Baseline Models for MultiNLI Corpus

This is the code used to establish baselines on the MultiNLI coprus introduced in [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/pdf/1704.05426.pdf).

## Models
- Continuous Bag of Words (CBOW):  in this model, each sentence is represented as the sum of the embedding representations of its
words. Main code for this model is in `models/cbow.py`
- Bi-directional LSTM: in this model, the average of the states of
a bidirectional LSTM RNN is used as the sentence representation. Main code for this model is in `models/bilstm.py`
- Enhanced Sequential Inference Model (ESIM): this uses Chen et al.'s (2017) ESIM model without ensembling with a TreeLSTM to create vector representation for sentences. Main code for this model is in `models/esim.py`

## Training schemes
- To train a model only on SNLI data, use `train_snli.py`. SNLI-dev accuracy is used to do early stopping. 
- To trian a model on either only MultiNLI or a mixture of MultiNLI and SNLI data, use `train_mnli.py`. The `alpha` flag determines what percentage of SNLI data is used in training. The default value is 0.0, which means the model will be only trained on MultiNLI data. If SNLI data is used by setting alpha to be greater than zero, a random sample is taken at the beginning of each epoch. Accuracy on MultiNLI's matched dev-set is used to do early stopping. 
- To train a model on a single MultiNLI genre or SNLI, use `train_genre.py`. To use this training scheme, you must call the `genre` flag and set it to a valid training genre (`travel`, `fiction`, `slate`, `telephone`, `government`, or `snli`). Accuracy on the dev-set for the chosen genre is used to do early stopping. Additionally, logs created with this training scheme contain evaulation statistics by genre. 

Required flags,

- `model_type`:
- `model_name`:

Optional flags,

- `datapath`:
- `ckptpath`:
- `logpath`:
- `emb_to_load`:
- `learning_rate`:
- `keep_rate`:
- `seq_length`:
- `emb_train`:
- `genre`:
- `alpha`:
- `test`:

Command template to run a training script,

`PYTHONPATH=$PYTHONPATH:. python training_script model_type experiment_name --keep_rate 0.5 --learning_rate 0.0004 --alpha 0.13 --emb_train`


Checkpoints: using tensorflow's inbuilt checkpointing apparatus using `tf.train.Saver`

- Most recent checkpoint
- Best checkpoint

Testing a model,

To test a trained model, simply call the `--test` flag to the command used for training. The best checkpoint will be loaded and used to evaluate the model's performance on the test-set.


## Analysis scripts
- Attention plots
- Learning curve plots
- Lookup checkpoints


