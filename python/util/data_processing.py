
import numpy as np
import re
import random
import json
import collections
import numpy as np

import parameters

FIXED_PARAMETERS = parameters.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

PADDING = "<PAD>"
#UNKNOWN = "<UNK>"


'''
Load NLI data
'''
def load_nli_data(path):
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            data.append(loaded_example)
        random.seed(1) ### Do we want to shuffle this data?
        random.shuffle(data)
    return data



'''
Annotate datasets with feature vectors.
'''
def sentences_to_padded_index_sequences(datasets):
    
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.lower().split()
    
    word_counter = collections.Counter()
    for i, dataset in enumerate(datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
        
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= padding:
                        index = word_indices[token_sequence[i - padding]]
                        #if token_sequence[i - padding] in word_indices:
                        #    index = word_indices[token_sequence[i - padding]]
                        #else:
                        #    index = word_indices[UNKNOWN]
                    else:
                        index = word_indices[PADDING]
                    example[sentence + '_index_sequence'][i] = index
    
    return indices_to_words, word_indices


'''
Load GloVe embeddings
'''

def loadEmebdding_zeros(path, word_indices):
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= FIXED_PARAMETERS["embeddings_to_load"]: 
                break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmebdding_rand(path, word_indices):
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    # Can do batch-wise
    #k = 10000
    for i in range(0, n, n):
        emb[i:i+n] = np.random.normal(size=(n,m))
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= FIXED_PARAMETERS["embeddings_to_load"]: 
                break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb

