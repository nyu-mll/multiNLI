def evaluate_classifier(classifier, eval_set, batch_size):
    correct = 0
    hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size
    #for i, example in enumerate(eval_set):
    for i in range(full_batch):
        hypothesis = hypotheses[i]
        #if hypothesis == example['label']:
        if hypothesis == eval_set[i]['label']:
            correct += 1        
    return correct / float(len(eval_set)), cost