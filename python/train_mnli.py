import tensorflow as tf
import os
import importlib
import random
from util import logger
import util.parameters as params
from util.data_processing import *
from util.evaluate import evaluate_classifier

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]
submodel = FIXED_PARAMETERS["model_subtype"]

module = importlib.import_module(".".join([model, submodel])) 
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we don't accidentally use different hyperparameter settings.
logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")
training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"])
dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"])
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"])

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])


logger.Log("Loading embeddings")
indices_to_words, word_indices = sentences_to_padded_index_sequences([training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli, test_matched, test_mismatched, test_snli])
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)

print np.shape(loaded_embeddings)

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.Log("Building model from %s.py" %(submodel))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.Log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()


    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels


    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):        
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 1
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_dev_mismat = 0.
        self.best_dev_snli = 0.
        self.best_train_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_epoch = 0


        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be passed as an argument.
        beta = int(self.alpha * len(train_snli))

        # Training data that will be used for evaluation,
        training_data_t = train_mnli + train_snli[:beta]
        random.shuffle(training_data_t)

        # Restore best-checkpoint if it exists
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                self.best_dev_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                self.best_dev_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_train_acc, train_cost = evaluate_classifier(self.classify, training_data_t[0:5000], self.batch_size)
                
                logger.Log("Restored best matched-dev acc: %f\n Restored best mismatched-dev acc: %f\n Restored best SNLI-dev acc: %f\n Restored best train acc: %f" %(self.best_dev_mat, self.best_dev_mismat, self.best_dev_snli, self.best_train_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)


        ### Training cycle
        logger.Log("Training...")
        logger.Log("Model will use %s percent of SNLI data during training" %(self.alpha * 100))

        while True:
            training_data = train_mnli + random.sample(train_snli, beta)
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels= self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: self.keep_rate}
                _, c = self.sess.run([self.optimizer, self.model.total_cost], feed_dict)

                # Since a single epoch can take a  ages, we'll print 
                # accuracy every 50 steps
                if self.step % self.display_step_freq == 0:
                    dev_acc_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                    dev_acc_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    train_acc, train_cost = evaluate_classifier(self.classify, training_data[0:5000], self.batch_size)

                    logger.Log("Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t Train acc: %f" %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, train_acc))
                    logger.Log("Step: %i\t Dev-matched cost %f\t Dev-mismatched cost %f\t Dev-SNLI cost %f\t Train cost %f" %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, train_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_dev_mismat = dev_acc_mismat
                        self.best_dev_snli = dev_acc_snli
                        self.best_train_acc = train_acc
                        self.best_epoch = self.epoch
                        logger.Log("Checkpointing with new best matched-dev accuracy: %f" %(self.best_dev_mat))

                self.step += 1

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the step
            # Evaluating only one batch worth of data -- simplifies implementation slightly
            if self.epoch % self.display_epoch_freq == 0:
                logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = train_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.epoch > self.best_epoch + 10):
                logger.Log("Best matched-dev accuracy: %s" %(self.best_dev_mat))
                logger.Log("Train accuracy: %s" %(self.best_train_acc))
                break

    def classify(self, examples):
        # This classifies a list of examples
        if examples in test_sets:
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.Log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(
                examples, self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0}
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1), cost


classifier = modelClassifier(FIXED_PARAMETERS["seq_length"])

# Now either train the model and then run it on the test set or just load the best checkpoint 
# and get accuracy on the test set. Default setting is to train the model.

test = params.train_or_test()
test_sets = [test_matched, test_mismatched, test_snli]

if test == False:
    classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])
else:
    logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Test acc on matched multiNLI: %s" %(evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.Log("Test acc on mismatched multiNLI: %s" %(evaluate_classifier(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
    #logger.Log("Test acc on SNLI: %s" %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])
