import tensorflow as tf
import parameters
from process import sentences_to_padded_index_sequences
from data_processing import *
from evaluate import evaluate_classifier 

FIXED_PARAMETERS = parameters.load_parameters()

training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])

loaded_embeddings = loadEmebdding(FIXED_PARAMETERS["embedding_data_path"], word_indices)


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

		# Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])


		## Define remaning parameters 
		self.E = tf.Variable(loaded_embeddings, trainable=False)

		self.W_rnn = {}
        self.W_r = {}
        self.W_z = {}
        self.b_rnn = {}
        self.b_r = {}
        self.b_z = {}
            
        for name in ['f', 'b']:
            in_dim = self.embedding_dim
            
            self.W_f[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_f[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_i[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_i[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_o[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_o[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))
        
            
        '''self.W_rnn['a'] = tf.Variable(tf.random_normal([self.dim + self.dim + self.dim, self.dim], stddev=0.1))
                                self.b_rnn['a'] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))
                        
                                self.W_r['a'] = tf.Variable(tf.random_normal([self.dim + self.dim + self.dim, self.dim], stddev=0.1))
                                self.b_r['a'] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))
                        
                                self.W_c['a'] = tf.Variable(tf.random_normal([self.dim + self.dim + self.dim, self.dim], stddev=0.1))
                                self.b_c['a'] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))
                                    
                                self.W_a_attn = tf.Variable(tf.random_normal([self.dim + self.dim, self.dim], stddev=0.1))'''
        
        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        
        
        
        # Define the LSTM function
        def lstm(emb, h_prev, c_prev, name):
            emb_h_prev = tf.concat(1, [emb, h_prev], name=name + '_emb_h_prev')
            f_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_f[name])  + self.b_f[name])
            i_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_i[name])  + self.b_i[name])
            c_tilde = tf.nn.tanh(tf.matmul(emb_h_prev, self.W_c[name])  + self.b_c[name])
            c = f_t * c_prev + i_t * c_tilde
            o_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_o[name])  + self.b_o[name])
            h = o_t * tf.nn.tanh(c)
            return h, c
        
        # Define one step of the premise encoder RNN
        def forward_step(x, h_prev, c_prev):
            emb = tf.nn.embedding_lookup(self.E, x)
            return lstm(emb, h_prev, c_prev, 'f')
        
        # Define one step of the hypothesis encoder RNN
        def backward_step(x, h_prev, c_prev):
            emb = tf.nn.embedding_lookup(self.E, x)
            return lstm(emb, h_prev, c_prev, 'b')

        # Split up the inputs into individual tensors
        self.x_premise_slices = tf.split(1, self.sequence_length, self.premise_x)
        self.x_hypothesis_slices = tf.split(1, self.sequence_length, self.hypothesis_x)
        
        self.h_zero = tf.zeros(tf.pack([tf.shape(self.premise_x)[0], self.dim]))
        
        #premise_h_prev_f = self.h_zero
        #premise_steps_list_f = []
        #hypothesis_h_prev = self.h_zero_hypothesis
        #hypothesis_steps_list = []

        premise_h_prev = {}
        premise_steps_list = {}
        hypothesis_h_prev = {}
        hypothesis_steps_list = {}
        premise_steps = {}
        hypothesis_steps = {}

        for name in ['f', 'b']:
        	premise_h_prev[name] = self.h_zero
        	premise_steps_list[name] = []
        	hypothesis_h_prev[name] = self.h_zero
        	hypothesis_steps_list[name] = []

        # Unroll FORWARD pass of both LSTMs for both sentences
        for t in range(self.sequence_length):
            a_t = tf.reshape(self.x_premise_slices[t], [-1])
            premise_h_prev['f'] = forward_step(a_t, premise_h_prev['f'])
            premise_steps_list['f'].append(premise_h_prev['f'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['f'] = hypothesis_step(b_t, hypothesis_h['f'])
            hypothesis_steps_list['f'].append(hypothesis_h_prev['f'])
            
        premise_steps['f'] = tf.pack(premise_steps_list['f'], axis=1)
        hypothesis_steps['f'] = tf.pack(premise_steps_list['f'], axis=1)
        
        '''# Unroll forward pass of premise LSTM
                                for t in range(self.sequence_length):
                                    x_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
                                    hypothesis_h_prev['f'] = hypothesis_step(x_t, hypothesis_h['f'])
                                    hypothesis_steps_list['f'].append(hypothesis_h_prev['f'])
                                    
                                hypothesis_steps['f'] = tf.pack(premise_steps_list['f'], axis=1, name='hypothesis_steps')'''

        # Unroll BACKWARD pass of both LSTMs for both sentences
        for t in range(self.sequence_length, -1, -1):
            a_t = tf.reshape(self.x_premise_slices[t], [-1])
            premise_h_prev['b'] = forward_step(a_t, premise_h_prev['b'])
            premise_steps_list['b'].append(premise_h_prev['b'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['b'] = hypothesis_step(b_t, hypothesis_h['b'])
            hypothesis_steps_list['b'].append(hypothesis_h_prev['b'])
            
        premise_steps['b'] = tf.pack(premise_steps_list['b'], axis=1)    
        hypothesis_steps['b'] = tf.pack(premise_steps_list['b'], axis=1)

        premise_list_bi = tf.concat(1, [premise_steps_list['f'], premise_steps_list['b']])
        hypothesis_list_bi = tf.concat(1, [hypothesis_steps_list['f'], hypothesis_steps_list['b']])

        premise_steps_bi = tf.concat(1, [premise_steps['f'], premise_steps['b']])
        hypothesis_steps_bi = tf.concat(1, [hypothesis_steps['f'], hypothesis_steps['b']])

        ### END BILISTM ###

        ### BEGIN ATTENTION: in progress      
		score_k_list = []
		score_j_list = []

		for j in range(len(premise_list_bi)):
		    score_k = tf.reduce_sum(tf.mul(premise_list_bi[s], concat_prev), 1, keep_dims=True)
		    score_k_list.append(score_kj)

		for k in range(len(hypothesis_list_bi)):
		    score_j = tf.reduce_sum(tf.mul(premise_list_bi[s], concat_prev), 1, keep_dims=True)
		    score_j_list.append(score_kj)

		score_k_all = tf.pack(score_k_list, axis=1)
		score_j_all = tf.pack(score_j_list, axis=1)
		alpha_k = tf.nn.softmax(score_k_all, dim=1)
		alpha_j = tf.nn.softmax(score_j_all, dim=1)          
		premise_attn_k = tf.reduce_sum(tf.mul(alpha_k, premise_steps_bi), 1)
		hypothesis_attn_j = tf.reduce_sum(tf.mul(alpha_j, hypothesis_steps_bi), 1)

		return premise_attn_k, hypothesis_attn_k
        
        #self.complete_attn_weights = tf.pack(alpha_kj_list, 2)

        ##### ATTENTION END


		# Get prediction
		self.logits = tf.matmul(self.h, self.W_cl) + self.b_cl

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
	            print "Epoch:", (epoch+1), "Cost:", avg_cost, \
	                "Dev acc:", evaluate_classifier(self.classify, dev_data[0:1000]), \
	                "Train acc:", evaluate_classifier(self.classify, training_data[0:1000])  
    
	def classify(self, examples):
	    # This classifies a list of examples
	    premise_vectors = np.vstack([example['sentence1_binary_parse_index_sequence'] for example in examples])
	    hypothesis_vectors = np.vstack([example['sentence2_binary_parse_index_sequence'] for example in examples])
	    logits = self.sess.run(self.logits, feed_dict={self.premise_x: premise_vectors,
	                                                   self.hypothesis_x: hypothesis_vectors})
	    return np.argmax(logits, axis=1)


classifier = CBOWClassifier(len(word_indices), FIXED_PARAMETERS["seq_length"])
classifier.train(training_set, dev_set)

evaluate_classifier(classifier.classify, dev_set)

