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

# Print fixed parameters, only print if this is a new log file 
# (don't need repeated information if we're picking up from an old checkpoint/log file)
if os.path.exists(logpath) == False:
    logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)

training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])
loaded_embeddings = loadEmebdding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

class EBIMClassifier:
    def __init__(self, vocab_size, seq_length):
        ## Define hyperparameters
        self.learning_rate = 0.0004
        self.training_epochs = 100
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"] # 300
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"] # 300
        self.batch_size = FIXED_PARAMETERS["batch_size"] #32
        self.keep_rate = FIXED_PARAMETERS["keep_rate"] # 0.5 
        self.sequence_length = FIXED_PARAMETERS["seq_length"] # 25

        # Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        # Define parameters
        self.E = tf.Variable(loaded_embeddings, trainable=True)

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
                
        
        # Define the LSTM call
        def lstm(emb, h_prev, c_prev, name): #removed name entry from function
            emb_h_prev = tf.concat([emb, h_prev], 1)
            f_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_f[name])  + self.b_f[name])
            i_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_i[name])  + self.b_i[name])
            c_tilde = tf.nn.tanh(tf.matmul(emb_h_prev, self.W_c[name])  + self.b_c[name])
            c = f_t * c_prev + i_t * c_tilde
            o_t = tf.nn.sigmoid(tf.matmul(emb_h_prev, self.W_o[name])  + self.b_o[name])
            h = o_t * tf.nn.tanh(c)
            return h, c

        # Embedding lookup and dropout at embedding layer
        def lstm_emb(x, h_prev, c_prev, name):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return lstm(emb_drop, h_prev, c_prev, name)

        # Full run of premise and hypothesis LSTM in one direction
        def lstm_run(inputs_prem, inputs_hyp, name, emb=True):
            prem_list = []
            hyp_list = []
            prem_h_prev = self.h_zero
            prem_c_prev = self.h_zero
            hyp_h_prev = self.h_zero
            hyp_c_prev = self.h_zero
            for t in range(self.sequence_length):
                if emb==True:
                    a_t = tf.reshape(inputs_prem[t], [-1])
                    b_t = tf.reshape(inputs_hyp[t], [-1])
                    prem_h_prev, prem_c_prev = lstm_emb(a_t, prem_h_prev, prem_c_prev, name)
                    hyp_h_prev, hyp_c_prev = lstm_emb(b_t, hyp_h_prev, hyp_c_prev, name)
                else:
                    a_t = inputs_prem[t]
                    b_t = inputs_hyp[t]
                    prem_h_prev, prem_c_prev = lstm(a_t, prem_h_prev, prem_c_prev, name)
                    hyp_h_prev, hyp_c_prev = lstm(b_t, hyp_h_prev, hyp_c_prev, name)

                prem_list.append(prem_h_prev)
                hyp_list.append(hyp_h_prev)

            return tf.stack(prem_list, axis=1), tf.stack(hyp_list, axis=1)

        def length(sentence):
            populated = tf.sign(tf.abs(sentence))
            length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
            return length

        # Get lengths of unpadded sentences
        prem_seq_lengths = length(self.premise_x)
        hyp_seq_lengths = length(self.hypothesis_x)


        ### First biLSTM layer ###

        # Split up the inputs into individual tensors
        self.x_premise_slices = tf.split(self.premise_x, self.sequence_length, 1)
        self.x_hypothesis_slices = tf.split(self.hypothesis_x, self.sequence_length, 1)

        self.x_premise_slices_back = tf.reverse_sequence(self.x_premise_slices, prem_seq_lengths, seq_axis=0, batch_axis=1)
        self.x_hypothesis_slices_back = tf.reverse_sequence(self.x_hypothesis_slices, hyp_seq_lengths, seq_axis=0, batch_axis=1)
        
        self.h_zero = tf.zeros(tf.stack([tf.shape(self.premise_x)[0], self.dim]))

        premise_f, hypothesis_f = lstm_run(self.x_premise_slices, self.x_hypothesis_slices, 'f')
        premise_rev, hypothesis_rev = lstm_run(self.x_premise_slices_back, self.x_hypothesis_slices_back, 'b')

        premise_b = tf.reverse_sequence(premise_rev, prem_seq_lengths, seq_axis=1, batch_axis=0)
        hypothesis_b = tf.reverse_sequence(hypothesis_rev, hyp_seq_lengths, seq_axis=1, batch_axis=0)

        premise_bi = tf.concat([premise_f, premise_b], axis=2)
        hypothesis_bi = tf.concat([hypothesis_f, hypothesis_b], axis=2) 

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)

        
        ### Attention ###

        scores_all = []
        premise_attn = []
        for i in range(self.sequence_length):
            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = tf.nn.softmax(scores_i, dim=1)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)

        scores_stack = tf.stack(scores_all, axis=2)

        hypothesis_attn = []
        for j in range(self.sequence_length):
            scores_j = tf.unstack(scores_stack, axis=2)[j]
            beta_j = tf.nn.softmax(scores_j, dim=1)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)
        
        #self.complete_attn_weights = stack lists of alpha_is and beta_js


        ### Subcomponent Inference ###

        m_a = []
        m_b = []
        
        for i in range(self.sequence_length):
            m_a_diff = premise_attn[i] - premise_list[i]
            m_a_mul = premise_attn[i] * premise_list[i]
            m_b_diff = hypothesis_attn[i] - hypothesis_list[i]
            m_b_mul = hypothesis_attn[i] * hypothesis_list[i]
            m_a_i = tf.concat([premise_list[i], premise_attn[i], m_a_diff, m_a_mul], 1)
            m_b_i = tf.concat([hypothesis_list[i], hypothesis_attn[i], m_b_diff, m_b_mul], 1)
            m_a.append(m_a_i)
            m_b.append(m_b_i)


        ### Inference Composition ###

        v1_f, v2_f = lstm_run(m_a, m_b, 'f2', emb=False)
        v1_rev, v2_rev = lstm_run(m_a, m_b, 'b2', emb=False)

        v1_b = tf.reverse_sequence(v1_rev, prem_seq_lengths, seq_axis=1, batch_axis=0)
        v2_b = tf.reverse_sequence(v2_rev, prem_seq_lengths, seq_axis=1, batch_axis=0)

        v1_bi = tf.concat([v1_f, v1_b], axis=2)
        v2_bi = tf.concat([v2_f, v2_b], axis=2)


        ### Pooling Layer ###

        v_1_ave = tf.reduce_sum(v1_bi, 1) / self.sequence_length
        v_2_ave = tf.reduce_sum(v2_bi, 1) / self.sequence_length
        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)


        ### TreeLSTM
        '''
        TODO: Build pseudo-treeLSTM and run it through all subsequent functions 
        '''

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.total_cost)

        # tf things: initialize variables and create placeholder for session
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
            indices = range(start_index, end_index)
            premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
            hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
            labels = [dataset[i]['label'] for i in indices]
            return premise_vectors, hypothesis_vectors, labels

    
    def train(self, training_data, dev_data):        
        self.sess = tf.Session()
        self.sess.run(self.init)

        # Restore best-checkpoint if it exists
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_acc = evaluate_classifier(self.classify, dev_data, self.batch_size)
                self.best_train_acc = evaluate_classifier(self.classify, training_data[0:5000], self.batch_size)
                logger.Log("Restored best dev acc: %f\t Restored best train acc: %f" %(self.best_dev_acc, self.best_train_acc))
            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)

        self.step = 1
        self.epoch = 0
        self.best_dev_acc = 0.
        self.best_train_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_epoch = 0

        ### Training cycle
        print 'Training...'

        while True:
        #for epoch in range(self.training_epochs):
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                _, c = self.sess.run([self.optimizer, self.total_cost], 
                                     feed_dict={self.premise_x: minibatch_premise_vectors,
                                                self.hypothesis_x: minibatch_hypothesis_vectors,
                                                self.y: minibatch_labels, 
                                                self.keep_rate_ph: self.keep_rate})

                # Since a single epoch can take a  ages, we'll print accuracy every
                # 250 steps as well as every epoch
                if self.step % self.display_step_freq == 0:
                    dev_acc = evaluate_classifier(self.classify, dev_data, self.batch_size)
                    train_acc = evaluate_classifier(self.classify, training_data[0:5000], self.batch_size)
                    logger.Log("Step: %i\t Dev acc: %f\t Train acc: %f" %(self.step, dev_acc, train_acc))

                if self.step % 1000 == 0:
                    self.saver.save(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt")
                    best_test = 100 * (1 - self.best_dev_acc / dev_acc)
                    if best_test > 0.1:
                        self.saver.save(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best")
                        self.best_dev_acc = dev_acc
                        self.best_train_acc = train_acc
                        self.best_epoch = self.epoch
                                  
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
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.epoch > self.best_epoch + 10):
                logger.Log("Best dev accuracy: %s" %(self.best_dev_acc))
                logger.Log("Train accuracy: %s" %(self.best_train_acc))
                break
    
    def classify(self, examples):
        # This classifies a list of examples
        if examples == test_set:
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best")

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors,minibatch_labels = self.get_minibatch(
                examples, self.batch_size * i, self.batch_size * (i + 1))
            logit = self.sess.run(self.logits, 
                                feed_dict={self.premise_x:minibatch_premise_vectors, 
                                self.hypothesis_x: minibatch_hypothesis_vectors, 
                                self.keep_rate_ph: 1.0})
            logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1)


classifier = EBIMClassifier(len(word_indices), FIXED_PARAMETERS["seq_length"])

# Now either train the model and then run it on the test set or just load the best checkpoint 
# and get accuracy on the test set. Default setting is to train the model.
test = parameters.train_or_test()

if test == False:
    classifier.train(training_set, dev_set)
    logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set, FIXED_PARAMETERS["batch_size"])))
else:
    logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set, FIXED_PARAMETERS["batch_size"])))

