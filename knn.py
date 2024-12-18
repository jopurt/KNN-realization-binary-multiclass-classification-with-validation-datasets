import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                manhat_dist = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
                dists[i_test][i_train] = manhat_dist
        # print(dists.shape)
        # print('dists',dists)
        return dists
    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # print('zzz',X[i_test],X.shape)
            # print('vvv',self.train_X,self.train_X.shape)
            manhat_dist = np.sum(np.abs(X[i_test] - self.train_X), axis=1)
            # Fill the whole row of dists[i_test]
            dists[i_test, :] = manhat_dist
            # print(manhat_dist[0])
            # print('ooo',manhat_dist,manhat_dist.shape)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        manhat_dist = np.abs(X[:, np.newaxis, :] - self.train_X).sum(axis=2)
        dists = manhat_dist
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # Implement choosing best class based on k
            # nearest training samples
            # print('num_test',num_test)
            # print('dists.shape',dists.shape)
            # print('dists',dists)
            # print('len(dists[i])',len(dists[i]))
            # print('dists[i]',dists[i])
            # print('len(self.train_y)',len(self.train_y))
            # print('self.train_y',self.train_y)
            # print('----------------------------------------------------')

            # Найти индексы k ближайших соседей
            # print('dists[i]', dists[i])
            # print('dists[i]',np.argsort(dists[i]))
            # print('min',dists[i][np.argsort(dists[i])[:self.k]])
            # print('min',dists[0][74])

            indices = np.argsort(dists[i])[:self.k]

            # print('indeces',indices)

            # Получить метки классов соседей
            # Get labels of nearest classes
            nearest_labels = self.train_y[indices]

            # print('nearest_labels',nearest_labels)

            # Определить наиболее частый класс (если равенство — взять минимальный)
            # Find the most popular class
            class_counts = np.bincount(nearest_labels)

            # print('class_counts',class_counts)
            # print(len(class_counts))
            # print(class_counts[0])

            pred[i] = np.argmax(class_counts)

            # print('pred[i]',pred[i])
            # print('---------------------------------------')

            pass
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, int)
        for i in range(num_test):
            # Implement choosing best class based on k
            # nearest training samples

            # index of nearest
            indices = np.argsort(dists[i])[:self.k]

            # label of nearest
            nearest_labels = self.train_y[indices]

            # uniques info
            uniques = np.unique(nearest_labels, return_counts=True)

            # num of occurrences
            num_of_occurrences = uniques[1]

            # index of max number
            index_of_max_nim = np.argmax(num_of_occurrences)

            # number of max occurances
            max_num = uniques[0][index_of_max_nim]

            # print('index of nearest:',indices, 'label of nearest:',nearest_labels,'uniques info:',uniques, 'num of occurrences:',num_of_occurrences,'max number:',max_num)

            pred[i] = max_num

            pass
        return pred
