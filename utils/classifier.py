"""
Osteoarthritis classification with SVM and Monte Carlo sampling
"""

import numpy as np
import pickle
import torch
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

from deformer.pca import SSM


def get_classification_data(data_path, label_path):
    """ Get data for classification experiment.

    Args:
        data_path (string): path to training_latents.pkl.
        label_path (string): path to classification labels.

    Returns:
        x (numpy array): N x D array of latent representation of N subject.
        y (numpy array): N labels.
    """
    data = torch.load(data_path, map_location="cpu")

    # Re-arrange level of detail latent vectors
    data_new1 = dict()
    data_new2 = dict()
    for key in data.keys():
        key_new = os.path.basename(key).split(".")[0]
        data_new1[key_new] = data[key][0].flatten().detach().numpy()
        data_new2[key_new] = data[key][1].flatten().detach().numpy()

    # Load labels
    with open(label_path, 'rb') as handle:
        osteoarthritis = pickle.load(handle)

    # Align dictionaries and get data
    keys = set.intersection(set(data_new1.keys()), set(osteoarthritis.keys()))

    # Compute PCA mode weights as latent representation
    x1 = np.stack([data_new1[key].flatten() for key in keys])
    x2 = np.stack([data_new2[key].flatten() for key in keys])
    pca1 = SSM(x1)
    pca2 = SSM(x2)
    x1 = pca1.get_weights(x1)
    x2 = pca2.get_weights(x2)
    x = np.concatenate((x1, x2), axis=1)

    y = np.array([int(osteoarthritis[key]) for key in keys])

    return x, y


def classify(x, y, n_splits=10000, train_sizes=None):
    """Classification experiment with stratified Monte Carlo sampling.

    Args:
        x (numpy array): PCA weights of the corresponding model.
        y (numpy array): targets.
        n_splits (int): number of Monte Carlo splits.
        independent_epsilon (bool): independent epsilon per point.
        train_sizes (numpy array): training set size percentages.

    Returns:
        results (dict): dictionary with entries of test set results 
            per partitioning.
    """
    # Initialize SVC classifier
    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=2000)
    results = {}

    # Default training set sizes
    if train_sizes is None:
        train_sizes = np.round(np.arange(0.9, 0., -0.1), 1)

        number_shapes = np.floor(len(x) * train_sizes)
        number_shapes //= 2
        number_shapes *= 2

        train_sizes = number_shapes / len(x)

    # Sample training sets per partitioning.
    for train_size in train_sizes:
        monte_carlo_split = StratifiedShuffleSplit(n_splits=n_splits,
                                                   train_size=train_size,
                                                   random_state=None)
        accurracy = np.zeros(n_splits)

        for n, (train_index, test_index) in enumerate(
                monte_carlo_split.split(x, y)):
            x_train = x[train_index]
            x_test = x[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            # Train and evaluate
            clf = LinearSVC().fit(x_train, y_train)
            accurracy[n] = clf.score(x_test, y_test)
        print(train_size, accurracy.mean())
        results[train_size] = accurracy

    return results
