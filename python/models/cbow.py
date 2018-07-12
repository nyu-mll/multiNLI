import tensorflow as tf

class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

		## Define placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define remaning parameters 
        # self.E = tf.Variable(embeddings, trainable=emb_train, name="emb")

        with tf.device('/cpu:0'):
            self.E = tf.Variable(tf.random_uniform(embeddings.shape, -1.0,1.0),
                        trainable=emb_train, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, embeddings.shape)
            self.embedding_init = self.E.assign(self.embedding_placeholder)



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

        ### MLP
        mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
        h_1 = tf.nn.relu(tf.matmul(mlp_input, self.W_0) + self.b_0)
        h_2 = tf.nn.relu(tf.matmul(h_1, self.W_1) + self.b_1)
        h_3 = tf.nn.relu(tf.matmul(h_2, self.W_2) + self.b_2)
        h_drop = tf.nn.dropout(h_3, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
