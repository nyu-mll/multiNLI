import tensorflow as tf
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
        self.learning_rate = 0.0004
        self.training_epochs = 100
        self.display_epoch_freq = 1
        self.display_step_freq = 250
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
            
        for name in ['f', 'b', 'f2', 'b2']:
            if name in ['f', 'b']:
                in_dim = self.embedding_dim
            else:
                in_dim = self.dim * 8
            
            self.W_f[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_f[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_i[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_i[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_o[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_o[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

            self.W_c[name] = tf.Variable(tf.random_normal([in_dim + self.dim, self.dim], stddev=0.1))
            self.b_c[name] = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
                
        
        # Define the LSTM function
        def lstm(emb, h_prev, c_prev, name): #removed name entry from function
            emb_h_prev = tf.concat(1, [emb, h_prev], name=name + '_emb_h_prev')
            f_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_f[name])  + self.b_f[name])
            i_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_i[name])  + self.b_i[name])
            c_tilde = tf.nn.tanh(tf.matmul(emb_h_prev, self.W_c[name])  + self.b_c[name])
            c = f_t * c_prev + i_t * c_tilde
            o_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_o[name])  + self.b_o[name])
            h = o_t * tf.nn.tanh(c)
            return h, c

        def lstm_step(x, h_prev, c_prev, name):
            emb = tf.nn.embedding_lookup(self.E, x)
            return lstm(emb, h_prev, c_prev, name)

        # Split up the inputs into individual tensors
        self.x_premise_slices = tf.split(1, self.sequence_length, self.premise_x)
        self.x_hypothesis_slices = tf.split(1, self.sequence_length, self.hypothesis_x)
        
        self.h_zero = tf.zeros(tf.pack([tf.shape(self.premise_x)[0], self.dim]))
        #print "HIDDEN ZERO", self.h_zero 

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
            premise_h_prev['f'], premise_c_prev['f'] = lstm_step(a_t, premise_h_prev['f'], premise_c_prev['f'], 'f')
            premise_steps_list['f'].append(premise_h_prev['f'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['f'], hypothesis_c_prev['f'] = lstm_step(b_t, hypothesis_h_prev['f'], hypothesis_c_prev['f'], 'f')
            hypothesis_steps_list['f'].append(hypothesis_h_prev['f'])
            
        premise_steps['f'] = tf.pack(premise_steps_list['f'], axis=1)
        hypothesis_steps['f'] = tf.pack(hypothesis_steps_list['f'], axis=1)

        # Unroll BACKWARD pass of LSTMs for both sentences
        for t in range(self.sequence_length-1 , -1, -1):
            a_t = tf.reshape(self.x_premise_slices[t], [-1])
            premise_h_prev['b'], premise_c_prev['b'] = lstm_step(a_t, premise_h_prev['b'], premise_c_prev['b'], 'b')
            premise_steps_list['b'].append(premise_h_prev['b'])

            b_t = tf.reshape(self.x_hypothesis_slices[t], [-1])
            hypothesis_h_prev['b'], hypothesis_c_prev['b']  = lstm_step(b_t, hypothesis_h_prev['b'], hypothesis_c_prev['b'], 'b')
            hypothesis_steps_list['b'].append(hypothesis_h_prev['b'])
        
        premise_list_bi = []
        hypothesis_list_bi = []

        for t in range(self.sequence_length):
            premise_bi_step = tf.concat(1, [premise_steps_list['f'][t], premise_steps_list['b'][t]])
            premise_list_bi.append(premise_bi_step)
            hypothesis_bi_step = tf.concat(1, [hypothesis_steps_list['f'][t], hypothesis_steps_list['b'][t]])
            hypothesis_list_bi.append(hypothesis_bi_step) 

        #print "PREMISE FORWARD LIST", premise_steps_list['f']
        #print "PREMISE LIST BI", premise_list_bi

        premise_steps_bi = tf.pack(premise_list_bi, axis=1)
        hypothesis_steps_bi = tf.pack(hypothesis_list_bi, axis=1)

        #print "PREMISE STEPS BI, PACKED", premise_steps_bi
        
        ### Attention ###

        scores_all = []
        premise_attn = []
        for i in range(len(premise_list_bi)):
            scores_i_list = []
            for j in range(len(hypothesis_list_bi)):
                #score_ij = tf.matmul(tf.transpose(premise_list_bi[i]), hypothesis_list_bi[j])
                score_ij = tf.reduce_sum(tf.mul(premise_list_bi[i], hypothesis_list_bi[i]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            scores_i = tf.pack(scores_i_list, axis=1)
            alpha_ij = tf.nn.softmax(scores_i, dim=1)
            a_tilde_i = tf.reduce_sum(tf.mul(alpha_ij, premise_steps_bi), 1)
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)

        #print "ONE ALPHA", alpha_ij
        #print "ONE A-TILDE", a_tilde_i 
        #print "test", tf.unpack(scores_all[0], axis=1)

        hypothesis_attn = []
        for j in range(len(hypothesis_list_bi)):
            scores_j_list = []
            for i in range(len(premise_list_bi)):
                score_ij = tf.unpack(scores_all[i], axis=1)[j]
                scores_j_list.append(score_ij)
            scores_j = tf.pack(scores_j_list, axis=1)
            beta_ij = tf.nn.softmax(scores_j, dim=1)
            b_tilde_j = tf.reduce_sum(tf.mul(beta_ij, hypothesis_steps_bi), 1)
            hypothesis_attn.append(b_tilde_j)


        #print "SCORES ALL", scores_all
        #print "PREM ATTN", premise_attn
        #print "HYP ATTN", hypothesis_attn
        
        #self.complete_attn_weights = tf.pack(alpha_kj_list, 2)

        ### Subcomponent Inference ###

        m_a = []
        m_b = []
        
        for i in range(self.sequence_length):
            m_a_diff = premise_attn[i] - premise_list_bi[i]
            m_a_mul = premise_attn[i] * premise_list_bi[i]
            m_b_diff = hypothesis_attn[i] - hypothesis_list_bi[i]
            m_b_mul = hypothesis_attn[i] * hypothesis_list_bi[i]
            m_a_i = tf.concat(1, [premise_list_bi[i], premise_attn[i], m_a_diff, m_a_mul])
            m_b_i = tf.concat(1, [hypothesis_list_bi[i], hypothesis_attn[i], m_b_diff, m_b_mul])
            m_a.append(m_a_i)
            m_b.append(m_b_i)

        #print "M_b", m_b
        #print "M_a", m_a

        ### Inference Composition ###

        v1_steps_list = {}
        v1_h_prev = {}
        v1_c_prev = {}
        #v1_steps = {}
        v2_steps_list = {}
        v2_h_prev = {}
        v2_c_prev = {}
        #v2_steps = {}

        for name in ['f2', 'b2']:
            v1_steps_list[name] = []
            v1_h_prev[name] = self.h_zero
            v1_c_prev[name] = self.h_zero
            v2_steps_list[name] = []
            v2_h_prev[name] = self.h_zero
            v2_c_prev[name] = self.h_zero

        # Unroll FORWARD pass of LSTMs for both composition layers
        for t in range(self.sequence_length):
            v1_h_prev['f2'], v1_c_prev['f2'] = lstm(m_a[t], v1_h_prev['f2'], v1_c_prev['f2'], 'f2')
            v1_steps_list['f2'].append(v1_h_prev['f2'])

            v2_h_prev['f2'], v2_c_prev['f2'] = lstm(m_b[t], v2_h_prev['f2'], v2_c_prev['f2'], 'f2')
            v2_steps_list['f2'].append(v2_h_prev['f2'])

        #v1_steps['f2'] = tf.pack(v1_steps_list['f2'], axis=1)    
        #v2_steps['f2'] = tf.pack(v2_steps_list['f2'], axis=1)

        # Unroll BACKWARD pass of LSTMs for both composition layers
        for t in range(self.sequence_length-1, -1, -1):
            v1_h_prev['b2'], v1_c_prev['b2'] = lstm(m_a[t], v1_h_prev['b2'], v1_c_prev['b2'], 'b2')
            v1_steps_list['b2'].append(v1_h_prev['b2'])

            v2_h_prev['b2'], v2_c_prev['b2'] = lstm(m_b[t], v2_h_prev['b2'], v2_c_prev['b2'], 'b2')
            v2_steps_list['b2'].append(v2_h_prev['b2'])

        #v1_steps['b2'] = tf.pack(v1_steps_list['b2'], axis=1) #?need?
        #v2_steps['b2'] = tf.pack(v2_steps_list['b2'], axis=1) #?need?

        #v1_list_bi = tf.concat(1, [v1_steps_list['f2'], v1_steps_list['b2']]) 
        #v2_list_bi = tf.concat(1, [v2_steps_list['f2'], v2_steps_list['b2']]) 

        #v1_steps_bi = tf.concat(1, [v1_steps['f2'], v1_steps['b2']]) #?need?
        #v2_steps_bi = tf.concat(1, [v2_steps['f2'], v2_steps['b2']]) #?need?

        v1_list_bi = []
        v2_list_bi = []

        for t in range(self.sequence_length):
            v1_bi_step = tf.concat(1, [v1_steps_list['f2'][t], v1_steps_list['b2'][t]])
            v1_list_bi.append(v1_bi_step)
            v2_bi_step = tf.concat(1, [v2_steps_list['f2'][t], v2_steps_list['b2'][t]])
            v2_list_bi.append(v2_bi_step)

        v1_steps_bi = tf.pack(v1_list_bi, axis=1)
        v2_steps_bi = tf.pack(v2_list_bi, axis=1)

        #print "V1 STEPS BI, PACKED", v1_steps_bi

        ### Pooling Layer ###

        v_1_ave = tf.reduce_sum(v1_steps_bi, 1) / self.sequence_length
        v_2_ave = tf.reduce_sum(v2_steps_bi, 1) / self.sequence_length
        v_1_max = tf.reduce_max(v1_steps_bi, 1)
        v_2_max = tf.reduce_max(v2_steps_bi, 1)

        v = tf.concat(1, [v_1_ave, v_2_ave, v_1_max, v_2_max])

        # MLP layers
        self.h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Get prediction
        # softmax instead of linear layer?
        self.logits = tf.matmul(self.h_mlp, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y))

        # Perform gradient descent
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_cost)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.total_cost)

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
        self.step = 1
        self.epoch = 0

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

                # Since a single epoch can take a  ages, we'll print accuracy every 250 steps as well as every epoch
                if self.step % self.display_step_freq == 0:
                    print "Step:", self.step, "Dev acc:", evaluate_classifier(self.classify, dev_data[:]), \
                        "Train acc:", evaluate_classifier(self.classify, training_data[0:5000]) 
                                  
                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the step
            # Evaluating only one batch worth of data -- simplifies implementation slightly
            #if (epoch+1) % self.display_epoch_freq == 0:
            if self.epoch % self.display_epoch_freq == 0:
                print "Epoch:", (epoch+1), "Cost:", avg_cost
            self.epoch += 1 
    
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

