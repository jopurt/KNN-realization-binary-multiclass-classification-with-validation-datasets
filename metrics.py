import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i in range(len(ground_truth)):
        if ((ground_truth[i] == True) & (prediction[i] == True)):
            true_positives += 1

        elif ((ground_truth[i] == False) & (prediction[i] == True)):
            false_positives += 1

        elif ((ground_truth[i] == False) & (prediction[i] == False)):
            true_negatives += 1

        elif ((ground_truth[i] == True) & (prediction[i] == False)):
            false_negatives += 1

    # print('true_positives',true_positives)
    # print('false_positives', false_positives)
    # print('true_negatives', true_negatives)
    # print('false_negatives', false_negatives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # Implement computing accuracy
    predicted = 0
    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i]:
            predicted += 1

    accuracy = predicted / len(ground_truth)
    # print(predicted)
    # print('real',ground_truth)
    # print('pred',prediction)
    return accuracy
