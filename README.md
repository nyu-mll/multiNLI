# Baseline Models for MultiNLI Corpus

This is the code used to establish baselines on the MultiNLI coprus introduced in [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/pdf/1704.05426.pdf)

## Models
- Continuous Bag of Words (CBOW):  in this model, each sentence is represented as the sum of the embedding representations of its
words. Main code for this model is in `models/cbow.py`
- Bi-directional LSTM: in this model, the average of the states of
a bidirectional LSTM RNN is used as the sentence representation. Main code for this model is in `models/bilstm.py`
- Enhanced Sequential Inference Model (ESIM): this uses Chen et al.'s (2017) ESIM model without ensembling with a TreeLSTM to create vector representation for sentences. Main code for this model is in `models/esim.py`

## Training schemes
- train_snli.py
- train_mnli.py
- train_genre.py

Optional flags,

- datapath:
- ckptpath:
- logpath:
- emb\_to\_load:
- learning\_rate:
- keep\_rate:
- seq\_length:
- emb\_train:
- genre:
- alpha:
- test:

Sample command template to run a training script,

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


