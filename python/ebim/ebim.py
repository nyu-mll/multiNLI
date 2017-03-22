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
        
        
        # Define the LSTM call

        ### Define biLSTM
        # Embedding lookup and dropout at embedding layer
        def emb_drop(x):
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

        ### First biLSTM layer ###

        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = biLSTM(premise_in, prem_seq_lengths, 'premise')
        hypothesis_outs, c2 = biLSTM(hypothesis_in, hyp_seq_lengths, 'hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)
        

        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        # can make loops over prem_seq and hyp_seq. premise_attn then is lenght of 
        # prem_seq. make rnag(seq_lenght - prem_seq) prem_attn = prem_bi
        for i in range(self.sequence_length):
            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            scores_i = tf.stack(scores_i_list, axis=1)
            # masked softmax?
            alpha_i = tf.nn.softmax(scores_i, dim=1)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)

        hypothesis_attn = []
        betas = []
        for j in range(self.sequence_length):
            scores_j = tf.unstack(scores_stack, axis=1)[j]
            # masked softmax?
            beta_j = tf.nn.softmax(scores_j, dim=1)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premimse_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots, 
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2)



        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)



        ### Inference Composition ###

        v1_outs, c3 = biLSTM(m_a, prem_seq_lengths, 'v1')
        v2_outs, c4 = biLSTM(m_b, hyp_seq_lengths, 'v2')

        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)



        ### Pooling Layer ###

        # print self.v1_bi and v1_sum. over batch.

        v_1_ave = tf.reduce_sum(v1_bi, 1) / self.sequence_length
        v_2_ave = tf.reduce_sum(v2_bi, 1) / self.sequence_length
        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)


        ### TreeLSTM ###
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
