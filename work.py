import numpy as np
import matplotlib.pyplot as plt
import time
import timeit


from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy

train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)


def show_nums():
    samples_per_class = 10  # Number of samples per class to visualize
    plot_index = 1
    for example_index in range(samples_per_class):
        for class_index in range(10):
            plt.subplot(10, 10, plot_index)
            image = train_X[train_y == class_index][example_index]
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            plot_index += 1
    plt.show()
# show_nums()

# print('train X\n',train_X)
# print('train y\n',train_y)
# print('test X\n',test_X)
# print('test y\n',test_y)

# First, let's prepare the labels and the source data

# Only select 0s and 9s
binary_train_mask = (train_y == 0) | (train_y == 9)
# print(binary_train_mask)

binary_train_X = train_X[binary_train_mask]
# print(binary_train_X)

binary_train_y = train_y[binary_train_mask] == 0
# print(binary_train_y)


binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
# print(binary_train_X)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)
# print(binary_test_X[0])

# Create the classifier and call fit to train the model
# KNN just remembers all the data
# print(binary_train_X.shape)
# print(binary_train_y.shape)
# print(binary_test_X.shape)

knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)

dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# Lets look at the performance difference
repeats = 10

time_two_loops = timeit.timeit(
    stmt='knn_classifier.compute_distances_two_loops(binary_test_X)',
    setup='from __main__ import knn_classifier, binary_test_X',
    number=repeats
)
print(f"Two loops: {time_two_loops / repeats:.6f} sec ")

time_one_loop = timeit.timeit(
    stmt='knn_classifier.compute_distances_one_loop(binary_test_X)',
    setup='from __main__ import knn_classifier, binary_test_X',
    number=repeats
)
print(f"One loop: {time_one_loop / repeats:.6f} sec ")

time_no_loops = timeit.timeit(
    stmt='knn_classifier.compute_distances_no_loops(binary_test_X)',
    setup='from __main__ import knn_classifier, binary_test_X',
    number=repeats
)
print(f"No loops: {time_no_loops / repeats:.6f} sec ")

prediction = knn_classifier.predict(binary_test_X)

# print('pred',prediction)

# print('real',binary_test_y)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

knn_classifier_2 = KNN(k=2)
knn_classifier_2.fit(binary_train_X, binary_train_y)
prediction = knn_classifier_2.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print('Binary classification:')
print("KNN with k = %s" % knn_classifier_2.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

# --- Cross-Validation ---

# Find the best k using cross-validation based on F1 score

def best_k_for_binary_classification():
    num_folds = 5
    train_folds_X = []
    train_folds_y = []

    num_samples = binary_train_X.shape[0]
    fold_size = num_samples // num_folds

    # 5 folds
    for i in range(num_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < num_folds - 1 else num_samples
        # print(start,end)
        train_folds_X.append(binary_train_X[start:end])
        train_folds_y.append(binary_train_y[start:end])

    # print(len(train_folds_X[0]))
    # print(len(train_folds_y[0]))

    k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)

    for k in k_choices:
        # Perform cross-validation
        # Go through every fold and use it for testing and all other folds for training
        # Perform training and produce F1 score metric on the validation dataset
        # Average F1 from all the folds and write it into k_to_f1

        f1_scores = []

        for i in range(num_folds):
            # Current fold for validation
            val_X = train_folds_X[i]
            val_y = train_folds_y[i]

            train_X_cv = np.vstack([train_folds_X[j] for j in range(num_folds) if j != i])
            train_y_cv = np.hstack([train_folds_y[j] for j in range(num_folds) if j != i])

            knn_classifier_valid = KNN(k=k)
            knn_classifier_valid.fit(train_X_cv, train_y_cv)

            # Predict on validation set
            val_predictions = knn_classifier_valid.predict(val_X)

            # print('pred',val_predictions)
            # print('real',val_y)

            precision, recall, f1, accuracy = binary_classification_metrics(val_predictions, val_y)
            f1_scores.append(f1)

        k_to_f1[k] = np.mean(f1_scores)
        pass

    print('Cross-validation for binary classification (5 validation sets)')

    for k in sorted(k_to_f1):
        print('k = %d, f1 = %f' % (k, k_to_f1[k]))

    best_k = max(k_to_f1, key=k_to_f1.get)
    max_f1 = k_to_f1[best_k]

    print('Best k:', best_k, 'F1 score:', max_f1)

    return best_k


#--- Let's look at best K on test data ---

best_k = best_k_for_binary_classification()

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(binary_train_X, binary_train_y)
prediction = best_knn_classifier.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("Best KNN with k = %s" % best_k)
print("Metrics on test data with best K (found using validation) Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))

# --- Now let's use all 10 classes ---

# Reshape to 1-dimensional array [num_samples, 32*32*3]
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)
# print(train_X)
# print(test_y)

knn_classifier = KNN(k=3)
knn_classifier.fit(train_X, train_y)

predict = knn_classifier.predict(test_X)
# print('pred',predict)
# print('real',test_y)

print('Multiclass classification:')

accuracy = multiclass_accuracy(predict, test_y)
print("Multiclass accuracy: %4.2f" % accuracy)

# Find the best K for multiclass classification using cross-validation based on accuracy
def best_k_for_multiclass_classification():
    num_folds = 5
    train_folds_X = []
    train_folds_y = []

    num_samples = train_X.shape[0]
    fold_size = num_samples // num_folds

    # 5 folds
    for i in range(num_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < num_folds - 1 else num_samples
        # print(start,end)
        train_folds_X.append(train_X[start:end])
        train_folds_y.append(train_y[start:end])

    k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    k_to_accuracy = {}

    for k in k_choices:
        # Perform cross-validation
        # Go through every fold and use it for testing and all other folds for validation
        # Perform training and produce accuracy metric on the validation dataset
        # Average accuracy from all the folds and write it into k_to_accuracy

        accuracy_scores = []

        for i in range(num_folds):
            # Current fold for validation
            val_X = train_folds_X[i]
            val_y = train_folds_y[i]

            train_X_cv = np.vstack([train_folds_X[j] for j in range(num_folds) if j != i])
            train_y_cv = np.hstack([train_folds_y[j] for j in range(num_folds) if j != i])

            knn_classifier_valid = KNN(k=k)
            knn_classifier_valid.fit(train_X_cv, train_y_cv)

            # Predict on validation set
            val_predictions = knn_classifier_valid.predict(val_X)

            # print('pred',val_predictions)
            # print('real',val_y)
            # print('val_predictions',val_predictions)
            accuracy = multiclass_accuracy(val_predictions, val_y)
            accuracy_scores.append(accuracy)

        k_to_accuracy[k] = np.mean(accuracy_scores)

        pass

    print('Cross-validation for multiclass classification (5 validation sets)')

    for k in sorted(k_to_accuracy):
        print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))

    best_k_multiclass = max(k_to_accuracy, key=k_to_accuracy.get)
    max_accuracy = k_to_accuracy[best_k_multiclass]

    print('Best K multiclass:', best_k_multiclass, 'Accuracy:', max_accuracy)

    return best_k_multiclass


# Set the best k as a best from computed
best_k_multiclass = best_k_for_multiclass_classification()

best_knn_classifier = KNN(k=best_k_multiclass)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)

# Accuracy should be around 20%!
accuracy = multiclass_accuracy(prediction, test_y)
print("Accuracy on test data with best K (found using validation): %4.2f" % accuracy)

# print('Try to find best K on test data (actually bad idea ^_^)')
# for k in [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]:
#     knn_classifier = KNN(k=k)
#     knn_classifier.fit(train_X, train_y)
#     test_predictions = knn_classifier.predict(test_X)
#     test_accuracy = multiclass_accuracy(test_predictions, test_y)
#     print(f"Test Accuracy with k={k}: {test_accuracy:.4f}")

