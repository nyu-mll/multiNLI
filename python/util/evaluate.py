def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1        
    return correct / float(len(eval_set)), cost

def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre,0) for genre in set(genres))
    count = dict((genre,0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print 'welp!'

    accuracy = {k: correct[k]/count[k] for k in correct}

    return accuracy, cost

