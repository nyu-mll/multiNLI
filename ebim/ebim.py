import tensorflow as tf
import numpy as np
import parameters
from data_processing import *
from evaluate import evaluate_classifier 

FIXED_PARAMETERS = parameters.load_parameters()

training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])

loaded_embeddings = loadEmebdding(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class EBIMClassifier:
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

        # Define parameters
        self.E = tf.Variable(loaded_embeddings, trainable=False)

        self.W_f = {}
        self.W_i = {}
        self.W_o = {}
        self.b_f = {}
        self.b_i = {}
        self.b_o = {}
        self.W_c = {}
        self.b_c = {}
            
        for name in ['f', 'b']:
            in_dim = self.embedding_dim
            
            self.W_f[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_f[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_i[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_i[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_o[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_o[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_c[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_c[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))
        
            
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
        def lstm(emb, h_prev, c_prev): #removed name entry from function
            emb_h_prev = tf.concat(1, [emb, h_prev], name=name + '_emb_h_prev')
            f_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_f[name])  + self.b_f[name])
            i_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_i[name])  + self.b_i[name])
            c_tilde = tf.nn.tanh(tf.matmul(emb_h_prev, self.W_c[name])  + self.b_c[name])
            c = f_t * c_prev + i_t * c_tilde
            o_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_o[name])  + self.b_o[name])
            h = o_t * tf.nn.tanh(c)
            return h, c
        
        '''# Define one step of the premise encoder RNN
                                def forward_step(x, h_prev, c_prev):
                                    emb = tf.nn.embedding_lookup(self.E, x)
                                    return lstm(emb, h_prev, c_prev, 'f')
                                
                                # Define one step of the hypothesis encoder RNN
                                def backward_step(x, h_prev, c_prev):
                                    emb = tf.nn.embedding_lookup(self.E, x)
                                    return lstm(emb, h_prev, c_prev, 'b')'''

        def lstm_step(x, h_prev, c_prev):
            emb = tf.nn.embedding_lookup(self.E, x)
            return lstm(emb, h_prev, c_prev)

        # Split up the inputs into individual tensors
        self.x_premise_slices = tf.split(1, self.sequence_length, self.premise_x)
        self.x_hypothesis_slices = tf.split(1, self.sequence_length, self.hypothesis_x)
        
        self.h_zero = tf.zeros(tf.pack([tf.shape(self.premise_x)[0], self.dim]))
        
        #premise_h_prev_f = self.h_zero
        #premise_steps_list_f = []
        #hypothesis_h_prev = self.h_zero_hypothesis
        #hypothesis_steps_list = []

        premise_h_prev = {}
        premise_c_prev = {}
        premise_steps_list = {}
        premise_steps = {}
        
        hypothesis_h_prev = {}
        hypothesis_c_prev = {}
        hypothesis_steps_list = {}
        hypothesis_steps = {}

        for name in ['f', 'b']:
            premise_h_prev[name] = self.h_zero
            premise_c_prev[name] = self.h_zero
            premise_steps_list[name] = []
            hypothesis_h_prev[name] = self.h_zero
            hypothesis_c_prev[name] = self.h_zero
            hypothesis_steps_list[name] = []

        # Unroll FORWARD pass of LSTMs for both sentences
        for t in range(self.sequence_length):
            a_t = tf.reshape(self.x_premise_slices[t], [-1])
            premise_h_prev['f'], premise_c_prev['f'] = lstm_step(a_t, premise_h_prev['f'], premise_c_prev['f'])
            premise_steps_list['f'].append(premise_h_prev['f'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['f'], hypothesis_c_prev['f'] = lstm_step(b_t, hypothesis_h_prev['f'], hypothesis_c_prev['f'])
            hypothesis_steps_list['f'].append(hypothesis_h_prev['f'])
            
        premise_steps['f'] = tf.pack(premise_steps_list['f'], axis=1)
        hypothesis_steps['f'] = tf.pack(hypothesis_steps_list['f'], axis=1)
        
        '''# Unroll forward pass of premise LSTM
                                for t in range(self.sequence_length):
                                    x_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
                                    hypothesis_h_prev['f'] = hypothesis_step(x_t, hypothesis_h['f'])
                                    hypothesis_steps_list['f'].append(hypothesis_h_prev['f'])
                                    
                                hypothesis_steps['f'] = tf.pack(premise_steps_list['f'], axis=1, name='hypothesis_steps')'''

        # Unroll BACKWARD pass of LSTMs for both sentences
        for t in range(self.sequence_length-1 , -1, -1):
            a_t = tf.reshape(self.x_premise_slices[t], [-1])
            premise_h_prev['b'], premise_c_prev['b'] = lstm_step(a_t, premise_h_prev['b'], premise_c_prev['b'])
            premise_steps_list['b'].append(premise_h_prev['b'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['b'], hypothesis_c_prev['b']  = lstm_step(b_t, hypothesis_h_prev['b'], hypothesis_c_prev['b'])
            hypothesis_steps_list['b'].append(hypothesis_h_prev['b'])
        
        premise_list_bi = []
        hypothesis_list_bi = []

        for t in range(self.sequence_length):
            premise_bi_step = tf.concat(0, [premise_steps_list['f'][t], premise_steps_list['b'][t]])
            premise_list_bi.append(premise_bi_step)
            hypothesis_bi_step = tf.concat(0, [hypothesis_steps_list['f'][t], hypothesis_steps_list['b'][t]])
            hypothesis_list_bi.append(hypothesis_bi_step) 

        premise_steps_bi = tf.pack(premise_list_bi, axis=1)
        hypothesis_steps_bi = tf.pack(hypothesis_list_bi, axis=1)

        ### Attention ###

        score_k_list = []
        score_j_list = []

        ### FIX ME reduce_sum needs to be one concat step with another concat step (not a list like right now)
        for j in range(len(premise_list_bi)):
            score_j = tf.reduce_sum(tf.mul(premise_list_bi[j], hypothesis_steps_bi), 1, keep_dims=True)
            score_j_list.append(score_j)

        for k in range(len(hypothesis_list_bi)):
            score_k = tf.reduce_sum(tf.mul(premise_list_bi[k], hypothesis_steps_bi), 1, keep_dims=True)
            score_k_list.append(score_k)

        # write above a nested for loops? -- (for j in seq: (for k in seq: (score= , score_matrix_jk= )))

        score_k_all = tf.pack(score_k_list, axis=1)
        score_j_all = tf.pack(score_j_list, axis=1)
        alpha_k = tf.nn.softmax(score_k_all, dim=1)
        alpha_j = tf.nn.softmax(score_j_all, dim=1)          
        premise_attn_k = tf.reduce_sum(tf.mul(alpha_k, premise_steps_bi), 1)
        hypothesis_attn_j = tf.reduce_sum(tf.mul(alpha_j, hypothesis_steps_bi), 1)
        
        #self.complete_attn_weights = tf.pack(alpha_kj_list, 2)

        ### Subcomponent Inference ###

        m_a = []
        m_b = []
        
        for i in range(seq_length):
            m_a_diff = premise_attn_k - premise_steps_bi
            m_a_mul = premise_attn_k * premise_steps_bi
            m_b_diff = hypothesis_attn_j - hypothesis_steps_bi
            m_b_mul = hypothesis_attn_j * hypothesis_steps_bi
            m_a_i = tf.concat(1, [premise_steps_bi, premise_attn_k, m_a_diff, m_a_mul])
            m_b_i = tf.concat(1, [hypothesis_steps_bi, hypothesis_attn_j, m_b_diff, m_b_mul])
            m_a.append(m_a_i)
            m_b.append(m_b_i)

        ### Inference Composition ###
        self.h_m_zero = tf.zeros(tf.pack([tf.shape(m_a)[0], self.dim])) ## ?? Not sure this is right dimension

        v1_steps_list = {}
        v1_h_prev = {}
        v1_c_prev = {}
        v1_steps = {}
        v2_steps_list = {}
        v2_h_prev = {}
        v2_c_prev = {}
        v2_steps = {}

        for name in ['f', 'b']:
            v1_steps_list[name] = []
            v1_h_prev[name] = self.h_m_zero
            v1_c_prev[name] = self.h_m_zero
            v2_steps_list[name] = []
            v2_h_prev[name] = self.h_m_zero
            v2_c_prev[name] = self.h_m_zero

        # Unroll FORWARD pass of LSTMs for both composition layers
        for t in range(self.sequence_length):
            v1_h_prev['f'], v1_c_prev['f'] = lstm(m_a[t], v1_h_prev['f'], v1_c_prev['f'])
            v1_steps_list['f'].append(v1_h_prev['f'])

            v2_h_prev['f'], v2_c_prev['f'] = lstm(m_b[t], v2_h_prev['f'], v2_c_prev['f'])
            v2_steps_list['f'].append(v2_steps_list['f'])

        v1_steps['f'] = tf.pack(v1_steps_list['f'], axis=1)    
        v2_steps['f'] = tf.pack(v2_steps_list['f'], axis=1)

        # Unroll BACKWARD pass of LSTMs for both composition layers
        for t in range(self.sequence_length-1, -1, -1):
            v1_h_prev['b'], v1_c_prev['b'] = lstm(m_a[t], v1_h_prev['b'], v1_c_prev['b'])
            v1_steps_list['b'].append(v1_h_prev['b'])

            v2_h_prev['b'], v2_c_prev['b'] = lstm(m_b[t], v2_h_prev['b'], v2_c_prev['b'])
            v2_steps_list['b'].append(v2_steps_list['b'])

        v1_steps['b'] = tf.pack(v1_steps_list['b'], axis=1) #?need?
        v2_steps['b'] = tf.pack(v2_steps_list['b'], axis=1) #?need?

        v1_list_bi = tf.concat(1, [v1_steps_list['f'], v1_steps_list['b']]) 
        v2_list_bi = tf.concat(1, [v2_steps_list['f'], v2_steps_list['b']]) 

        v1_steps_bi = tf.concat(1, [v1_steps['f'], v1_steps['b']]) #?need?
        v2_steps_bi = tf.concat(1, [v2_steps['f'], v2_steps['b']]) #?need?

        # Convert to fixed lenght vector


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


classifier = EBIMClassifier(len(word_indices), FIXED_PARAMETERS["seq_length"])
classifier.train(training_set, dev_set)

evaluate_classifier(classifier.classify, dev_set)

