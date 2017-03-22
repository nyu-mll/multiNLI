import tensorflow as tf

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=True)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))
        

        ## Define biLSTM
        # Embedding lookup and dropout at embedding layer
        def emb_drop(x):
            #emb = tf.nn.embedding_lookup(self.E, tf.transpose(x))
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        def length(sentence):
            populated = tf.sign(tf.abs(sentence))
            length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
            return length

        # Get lengths of unpadded sentences
        prem_seq_lengths = length(self.premise_x)
        hyp_seq_lengths = length(self.hypothesis_x)

        def biLSTM(inputs, seq_len, name):
            with tf.name_scope(name):
              with tf.variable_scope('forward' + name):
                lstm_fwd = tf.contrib.rnn.LSTMCell(self.dim, forget_bias=1.0)
              with tf.variable_scope('backward' + name):
                lstm_bwd = tf.contrib.rnn.LSTMCell(self.dim, forget_bias=1.0)

              hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

            return hidden_states, cell_states


        ### BiLSTM layer ###

        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = biLSTM(premise_in, prem_seq_lengths, 'premise')
        hypothesis_outs, c2 = biLSTM(hypothesis_in, hyp_seq_lengths, 'hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_final = tf.unstack(premise_bi, axis=1)[-1]
        hypothesis_final = tf.unstack(hypothesis_bi, axis=1)[-1]
        

        ### Mou et al. concat layer ###
        diff = tf.subtract(premise_final, hypothesis_final)
        mul = tf.multiply(premise_final, hypothesis_final)
        h = tf.concat([premise_final, hypothesis_final, diff, mul], 1)

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(h, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
