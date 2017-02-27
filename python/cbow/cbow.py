import tensorflow as tf
import os
from util import logger
import util.parameters
from util.data_processing import *
from util.evaluate import evaluate_classifier 

FIXED_PARAMETERS = parameters.load_parameters()

modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

# Print fixed parameters, only print if this is a new log file (don't need repeated information if we're picking up from an old checkpoint/log file)
if os.path.exists(logpath) == False:
	logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)

training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])
loaded_embeddings = loadEmebdding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

class CBOWClassifier:
	def __init__(self, vocab_size, seq_length):
		## Define hyperparameters
		self.learning_rate = FIXED_PARAMETERS["learning_rate"]
		#self.training_epochs = 100
		self.display_epoch_freq = 1
		self.display_step_freq = 250
		self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
		self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
		self.batch_size = FIXED_PARAMETERS["batch_size"]
		self.keep_rate = FIXED_PARAMETERS["keep_rate"]
		self.sequence_length = FIXED_PARAMETERS["seq_length"]

		## Define placeholders
		self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.y = tf.placeholder(tf.int32, [None])
		self.keep_rate_ph = tf.placeholder(tf.float32, [])


		## Define remaning parameters 
		self.E = tf.Variable(loaded_embeddings, trainable=True, name="emb")

		self.W_0 = tf.Variable(tf.random_normal([self.embedding_dim * 4, self.dim], stddev=0.1), name="w0")
		self.b_0 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b0")

		self.W_1 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w1")
		self.b_1 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b1")

		self.W_2 = tf.Variable(tf.random_normal([self.dim, self.dim], stddev=0.1), name="w2")
		self.b_2 = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name="b2")

		self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1), name="wcl")
		self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1), name="bcl")


		## Calculate representaitons by CBOW method
		emb_premise = tf.nn.embedding_lookup(self.E, self.premise_x) 
		emb_premise_drop = tf.nn.dropout(emb_premise, self.keep_rate_ph)
	
		emb_hypothesis = tf.nn.embedding_lookup(self.E, self.hypothesis_x)
		emb_hypothesis_drop = tf.nn.dropout(emb_hypothesis, self.keep_rate_ph)

		premise_rep = tf.reduce_sum(emb_premise_drop, 1)
		hypothesis_rep = tf.reduce_sum(emb_hypothesis_drop, 1)

		## Combinations
		h_diff = premise_rep - hypothesis_rep
		h_mul = premise_rep * hypothesis_rep

		### MLP HERE (without dropout)
		mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
		h_1 = tf.nn.relu(tf.matmul(mlp_input, self.W_0) + self.b_0)
		h_2 = tf.nn.relu(tf.matmul(h_1, self.W_1) + self.b_1)
		h_3 = tf.nn.relu(tf.matmul(h_2, self.W_2) + self.b_2)
		h_drop = tf.nn.dropout(h_3, self.keep_rate_ph)
		
		# Get prediction
		self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

		# Define the cost function
		self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

		# Perform gradient descent
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.total_cost)

		# tf things: initialize variables abd create palceholder for sesson
		self.init = tf.initialize_all_variables()
		self.sess = None
		self.saver = tf.train.Saver()
	
	def train(self, training_data, dev_data):
		def get_minibatch(dataset, start_index, end_index):
			indices = range(start_index, end_index)
			premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
			hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
			labels = [dataset[i]['label'] for i in indices]
			return premise_vectors, hypothesis_vectors, labels

		self.sess = tf.Session()
		self.sess.run(self.init)

		# Restore best-checkpoint if it exists
		ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
		if os.path.isfile(ckpt_file):
			self.saver.restore(self.sess, ckpt_file)
			#print("Model restored from file: %s" % ckpt_file)
			logger.Log("Model restored from file: %s" % ckpt_file)

		self.step = 1
		self.epoch = 0

		#print 'Training...'
		logger.Log("Training...")

		### Training Cycle

		self.best_dev_acc = 0.
		self.best_train_acc = 0.
		self.last_train_acc = [.001, .001, .001, .001, .001]

		while True:
		#for epoch in range(self.training_epochs):
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
												self.y: minibatch_labels,
												self.keep_rate_ph: self.keep_rate})

				if self.step % self.display_step_freq == 0:
					dev_acc = evaluate_classifier(self.classify, dev_data)
					train_acc = evaluate_classifier(self.classify, training_data[0:5000])
					logger.Log("Step: %i\t Dev acc: %f\t Train acc: %f" %(self.step, dev_acc, train_acc))

				if self.step % 10000 == 0:
					self.saver.save(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt")
					best_test = 100 * (1 - self.best_dev_acc / dev_acc)
					if best_test > 0.1:
						self.saver.save(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best")
						self.best_dev_acc = dev_acc
						self.best_train_acc = train_acc
								  
				self.step += 1

				# Compute average loss
				avg_cost += c / (total_batch * self.batch_size)
								
			# Display some statistics about the step
			# Evaluating only one batch worth of data -- simplifies implementation slightly
			if self.epoch % self.display_epoch_freq == 0:
				logger.Log("Epoch: %i\t Cost: %f" %(self.epoch+1, avg_cost))

			self.epoch += 1
			self.last_train_acc[(self.epoch % 5) - 1] = train_acc

			# Early stopping
			termination_test = 100 * (self.best_dev_acc / dev_acc - 1)
			progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

			if ((progress < 0.1) or 
				(self.epoch > 1000) or 
				(termination_test > 1.0)):
				logger.Log("Best dev accuracy: %s" %(self.best_dev_acc))
				logger.Log("Train accuracy: %s" %(self.best_train_acc))
				break 
	
	def classify(self, examples):#, using_best=False):
		# This classifies a list of examples
		if examples == test_set:
			self.sess = tf.Session()
			self.sess.run(self.init)
			self.saver.restore(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best")
		premise_vectors = np.vstack([example['sentence1_binary_parse_index_sequence'] for example in examples])
		hypothesis_vectors = np.vstack([example['sentence2_binary_parse_index_sequence'] for example in examples])
		logits = self.sess.run(self.logits, feed_dict={self.premise_x: premise_vectors,
													   self.hypothesis_x: hypothesis_vectors,
													   self.keep_rate_ph: 1.0})
		return np.argmax(logits, axis=1)


classifier = CBOWClassifier(len(word_indices), FIXED_PARAMETERS["seq_length"])

# Now either train the model and then run it on the test set or just load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
test = parameters.train_or_test()

if test == False:
	classifier.train(training_set, dev_set)
	logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set)))
else:
	logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set)))

