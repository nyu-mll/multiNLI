# Baseline Models for MultiNLI Corpus

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

Template command to run a training script,
`PYTHONPATH=$PYTHONPATH:. python training_script model_type experiment_name --keep_rate 0.5 --learning_rate 0.0004 --alpha 0.13 --emb_train`.

## Analysis scripts