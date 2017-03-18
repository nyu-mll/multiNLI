'''
Example command to run:
PYTHONPATH=. python analysis/attn_plots.py ebim ebim --datapath ./data --logpath ./logs
'''

import tensorflow as tf
import os
import python.util.parameters
from python.util.data_processing import *
from python.util.evaluate import evaluate_classifier
import matplotlib.pyplot as plt

FIXED_PARAMETERS = parameters.load_parameters()

if FIXED_PARAMETERS["model_type"] == 'ebim':
    from python.ebim.ebim_box import MyModel
else:
    assert (FIXED_PARAMETERS["model_type"] != 'ebim'), \
       'Give a valid model that uses attention.'

modname = FIXED_PARAMETERS["model_name"]

######################### LOAD DATA #############################

print "Loading data."
training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

print "Processing data."
indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])

print "Loading GloVe embeddings."
loaded_embeddings = loadEmebdding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 

        print "Loading model."
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_attn(self, examples):
        self.sess = tf.Session()
        self.sess.run(self.init)
        print "Restoring best checkpoint."
        self.saver.restore(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best")

        logits = np.empty(3)
        premise_vectors = np.vstack([example['sentence1_binary_parse_index_sequence'] for example in examples])
        hypothesis_vectors = np.vstack([example['sentence2_binary_parse_index_sequence'] for example in examples])

        print "Getting attention weights"
        feed_dict={self.model.premise_x: premise_vectors,
                                                self.model.hypothesis_x: hypothesis_vectors,
                                                self.model.keep_rate_ph: 1.0}
        alpha_weights, beta_weights, logit = self.sess.run([self.model.alpha_s, self.model.beta_s, self.model.logits],feed_dict=feed_dict)
        logits = np.vstack([logits, logit])
        return np.reshape(alpha_weights, [len(examples), 25, 25]), np.reshape(beta_weights, [len(examples), 25, 25]), np.argmax(logits[1:], axis=1)
        
    def plot_attn(self, examples):
        alphas, betas, prediction = self.get_attn(examples)

        for i in range(len(examples)):    
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.matshow(alphas[i,:,:], vmin=0., vmax=1., cmap=plt.cm.inferno)

            premise_tokens = [indices_to_words[index] for index in examples[i]['sentence1_binary_parse_index_sequence']]
            hypothesis_tokens = [indices_to_words[index] for index in examples[i]['sentence2_binary_parse_index_sequence']]

            ax.set_yticklabels(hypothesis_tokens)
            ax.set_xticklabels(premise_tokens, rotation=45)

            ax.set_xticks(np.arange(0, 25, 1.0))
            ax.set_yticks(np.arange(0, 25, 1.0))
            ax.tick_params(axis='both', labelsize=10)
            title_e = examples[i]
            true = title_e['label']
            ax.set_xlabel('True label: %s, Predicted label: %s' %(true, prediction[i]))
            plt.tight_layout()
            #plt.savefig("../attn_box/" + modname + str(2) + '_' + str(i) + ".png")
            plt.show()


classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

#assert (classifier.get_attn(training_set[0:2])[0, :, :] == \
#        classifier.get_attn(training_set[0:3])[0, :, :]).all(), \
#       'Warning: There is cross-example information flow.'

classifier.plot_attn(training_set[0:1])
