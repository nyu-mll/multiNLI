import tensorflow as tf
import util.parameters
from util.data_processing import *
from util.evaluate import evaluate_classifier 
from util import logger
import os

FIXED_PARAMETERS = parameters.load_parameters()

logger = logger.Logger(os.path.join(FIXED_PARAMETERS["log_path"], "cbow") + ".log")

training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])

loaded_embeddings = loadEmebdding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

class CBOWClassifier:
	def __init__(self, vocab_size, seq_length):
		## Define hyperparameters
		self.learning_rate = 0.03
		self.training_epochs = 100
		self.display_epoch_freq = 1
		self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
		self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
		self.batch_size = FIXED_PARAMETERS["batch_size"]
		#self.keep_rate = 0.5
		self.sequence_length = FIXED_PARAMETERS["seq_length"]

		## Define placeholders
		self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.y = tf.placeholder(tf.int32, [None])


		## Define remaning parameters 
		self.E = tf.Variable(loaded_embeddings, trainable=False)

		self.W_0 = tf.Variable(tf.random_normal([self.embedding_dim * 4, self.dim], stddev=0.1))
		self.b_0 = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

		self.W_1 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1))
		self.b_1 = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

		self.W_2 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1))
		self.b_2 = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

		self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
		self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))


		## Calculate representaitons by CBOW method
		emb_premise = tf.nn.embedding_lookup(self.E, self.premise_x) 
		emb_hypothesis = tf.nn.embedding_lookup(self.E, self.hypothesis_x)
			# expected shape: [None, sequence_length, embedding_size]

		premise_rep = tf.reduce_sum(emb_premise, 1)
		hypothesis_rep = tf.reduce_sum(emb_hypothesis, 1)
			# expected shape: [None, embedding_size]

		## Combinations
		h_diff = tf.sub(premise_rep, hypothesis_rep)
		h_mul = tf.mul(premise_rep, hypothesis_rep) 

		### MLP HERE (without dropout)
		mlp_input = tf.concat(1, [premise_rep, hypothesis_rep, h_diff, h_mul])
		h_1 = tf.nn.relu(tf.matmul(mlp_input, self.W_0) + self.b_0)
		h_2 = tf.nn.relu(tf.matmul(h_1, self.W_1) + self.b_1)
		self.h_3 = tf.nn.relu(tf.matmul(h_2, self.W_2) + self.b_2)

		# Get prediction
		self.logits = tf.matmul(self.h_3, self.W_cl) + self.b_cl

		# Define the cost function
		self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y))

		# Perform gradient descent
		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_cost)

		# tf things: initialize variables abd create palceholder for sesson
		self.init = tf.initialize_all_variables()
		self.sess = None
    
	def train(self, training_data, dev_data):
	    def get_minibatch(dataset, start_index, end_index):
	        indices = range(start_index, end_index)
	        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
	        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
	        labels = [dataset[i]['label'] for i in indices]
	        return premise_vectors, hypothesis_vectors, labels
	    
	    self.sess = tf.Session()
	    
	    self.sess.run(self.init)
	    print 'Training...'

	    # Training cycle
	    for epoch in range(self.training_epochs):
	        random.shuffle(training_data)
	        avg_cost = 0.
	        total_batch = int(len(training_data) / self.batch_size)
	        
	        # Loop over all batches in epoch
	        for i in range(total_batch):
	            # Assemble a minibatch of the next B examples
	            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = get_minibatch(
	                training_data, self.batch_size * i, self.batch_size * (i + 1))

	            # Run the optimizer to take a gradient step, and also fetch the value of the 
	            # cost function for logging
	            _, c = self.sess.run([self.optimizer, self.total_cost], 
	                                 feed_dict={self.premise_x: minibatch_premise_vectors,
	                                            self.hypothesis_x: minibatch_hypothesis_vectors,
	                                            self.y: minibatch_labels})

	            # Compute average loss
	            avg_cost += c / (total_batch * self.batch_size)
	                            
	        # Display some statistics about the step
	        # Evaluating only one batch worth of data -- simplifies implementation slightly
	        if (epoch+1) % self.display_epoch_freq == 0:
	            '''print "Epoch:", (epoch+1), "Cost:", avg_cost, \
	            	            	                	                "Dev acc:", evaluate_classifier(self.classify, dev_data[0:1000]), \
	            	            	                	                "Train acc:", evaluate_classifier(self.classify, training_data[0:1000])'''
	            logger.Log("Epoch: %i\t Cost: %f\t Dev acc: %f\t Train acc: %f" %(epoch+1, avg_cost, evaluate_classifier(self.classify, dev_data[:]), evaluate_classifier(self.classify, training_data[0:5000])))
    
	def classify(self, examples):
	    # This classifies a list of examples
	    premise_vectors = np.vstack([example['sentence1_binary_parse_index_sequence'] for example in examples])
	    hypothesis_vectors = np.vstack([example['sentence2_binary_parse_index_sequence'] for example in examples])
	    logits = self.sess.run(self.logits, feed_dict={self.premise_x: premise_vectors,
	                                                   self.hypothesis_x: hypothesis_vectors})
	    return np.argmax(logits, axis=1)


classifier = CBOWClassifier(len(word_indices), FIXED_PARAMETERS["seq_length"])
classifier.train(training_set, dev_set)

print "Test acc:", evaluate_classifier(classifier.classify, test_set)

