import tensorflow as tf
import os
import importlib
from util import logger
import util.parameters
from util.data_processing import *
from util.evaluate import evaluate_classifier

FIXED_PARAMETERS = parameters.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]
submodel = FIXED_PARAMETERS["model_subtype"]

module = importlib.import_module(".".join([model, submodel])) 
MyModel = getattr(module, 'MyModel')

# Print fixed parameters, only print if this is a new log file 
# (don't need repeated information if we're picking up from an old checkpoint/log file)
if os.path.exists(logpath) == False:
    logger.Log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)


######################### LOAD DATA #############################

logger.Log("Loading data")
training_set = load_nli_data(FIXED_PARAMETERS["training_data_path"])
dev_set = load_nli_data(FIXED_PARAMETERS["dev_data_path"])
test_set = load_nli_data(FIXED_PARAMETERS["test_data_path"])

logger.Log("Loading embeddings")
indices_to_words, word_indices = sentences_to_padded_index_sequences([training_set, dev_set, test_set])
loaded_embeddings = loadEmebdding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"] 

        logger.Log("Building model from %s.py" %(submodel))
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=loaded_embeddings)

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


    def train(self, training_data, dev_data):        
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 1
        self.epoch = 0
        self.best_dev_acc = 0.
        self.best_train_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_epoch = 0

        # Restore best-checkpoint if it exists
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_acc, dev_cost = evaluate_classifier(self.classify, dev_data, self.batch_size)
                self.best_train_acc, train_cost = evaluate_classifier(self.classify, training_data[0:5000], self.batch_size)
                logger.Log("Restored best dev acc: %f\t Restored best train acc: %f" %(self.best_dev_acc, self.best_train_acc))
            self.saver.restore(self.sess, ckpt_file)
            logger.Log("Model restored from file: %s" % ckpt_file)


        ### Training cycle
        logger.Log("Training...")

        while True:
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
                    dev_acc, dev_cost = evaluate_classifier(self.classify, dev_data, self.batch_size)
                    train_acc, train_cost = evaluate_classifier(self.classify, training_data[0:5000], self.batch_size)
                    logger.Log("Step: %i\t Dev acc: %f\t Train acc: %f\t Dev cost %f\t Train cost %f" %(self.step, dev_acc, train_acc, dev_cost, train_cost))

                if self.step % 500 == 0:
                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_acc / dev_acc)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_acc = dev_acc
                        self.best_train_acc = train_acc
                        self.best_epoch = self.epoch
                        logger.Log("Checkpointing with new best dev accuracy: %f" %(self.best_dev_acc))

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

test = parameters.train_or_test()

if test == False:
    classifier.train(training_set, dev_set)
    logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set, FIXED_PARAMETERS["batch_size"]))[0])
else:
    logger.Log("Test acc: %s" %(evaluate_classifier(classifier.classify, test_set, FIXED_PARAMETERS["batch_size"]))[0])

