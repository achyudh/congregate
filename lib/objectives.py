from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _counts(labels):
    """
    Calculates the count of documents in each cluster
    :param labels: cluster labels for all the documents
    :return:
    """
    nr = defaultdict(int)
    for label in labels:
        nr[label] += 1
    return nr


def _centroids(features, labels):
    """

    :param features: iterable of tf-idf features for every document
    :param labels: cluster labels for all the documents in features
    :return:
    """
    _cr = defaultdict(list)
    for di in range(len(features)):
        _cr[labels[di]].append(features[di])

    cr = dict()
    for label in _cr.keys():
        cr[label] = np.mean(np.array(_cr[label]), axis=0)

    return cr


def I1(features, labels):
    """
    Computes the sum of the average pairwise cosine similarities between the documents
    assigned to each cluster weighted according to the size of each cluster
    :param features: iterable of tf-idf features for every document
    :param labels: cluster labels for all the documents in features
    :return: objective value for the corresponding clustering
    """
    nr = _counts(labels)
    cos = cosine_similarity(features)

    objective_value = 0
    for cluster in nr.keys():
        inner_sum = 0
        for di in range(len(features)):
            for dj in range(di + 1, len(features)):
                if labels[di] == labels[dj] == cluster:
                    inner_sum += cos[di][dj]
        objective_value += (inner_sum / nr[cluster])

    return objective_value


def I2(features, labels):
    """
    Computes the cosine similarity between each document and the centroid of the cluster
    that is assigned to. Comparing I1 and I2 we see that the essential difference
    :param features: iterable of tf-idf features for every document
    :param labels: cluster labels for all the documents in features
    :return: objective value for the corresponding clustering
    """
    cr, cr_index = list(), dict()
    for cluster, centroid in _centroids(features, labels).items():
        cr_index[cluster] = len(cr)
        cr.append(centroid)
    cos = cosine_similarity(features, np.array(cr))

    objective_value = 0
    for cluster in cr_index.keys():
        inner_sum = 0
        for di in range(len(features)):
            if labels[di] == cluster:
                inner_sum += cos[di][cr_index[cluster]]
        objective_value += inner_sum

    return objective_value


def E1(features, labels):
    """
    Computes the cosine similarity between the centroid vector of each cluster and the
    centroid vector of the entire collection
    :param features: iterable of tf-idf features for every document
    :param labels: cluster labels for all the documents in features
    :return: objective value for the corresponding clustering
    """
    cr, cr_index = list(), dict()
    for cluster, centroid in _centroids(features, labels).items():
        cr_index[cluster] = len(cr)
        cr.append(centroid)

    nr = _counts(labels)
    gcr = [np.mean(np.array(features), axis=0)]
    cos = cosine_similarity(np.array(gcr), np.array(cr))

    objective_value = 0
    for cluster in cr_index.keys():
        objective_value += nr[cluster] * cos[0][cr_index[cluster]]

    return objective_value
