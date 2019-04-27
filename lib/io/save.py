import os

import numpy as np


def _nonzero(features):
    count = 0
    for document in features:
        for index in range(len(features)):
            if document[index] != 0:
                count += 1
    return count


def dense_matrix(features, labels, dataset):
    """

    :param features:
    :param labels:
    :param dataset:
    :return:
    """
    with open(os.path.join('data', '%s.mat' % dataset), 'w') as mat_file:
        mat_file.write('%d %d \n' % (features.shape[0], features.shape[1]))
        for document in features:
            mat_file.write(' '.join(str(x) for x in document))
            mat_file.write('\n')

    with open(os.path.join('data', '%s.mat.rclass' % dataset), 'w') as rclass_file:
        for label in labels:
            rclass_file.write(str(label) + '\n')


def sparse_matrix(features, labels, dataset):
    """

    :param features:
    :param labels:
    :param dataset:
    :return:
    """
    with open(os.path.join('data', '%s.sparse.mat' % dataset), 'w') as mat_file:
        mat_file.write('%d %d %d \n' % (features.shape[0], features.shape[1], _nonzero(features)))
        for document in features:
            first_column = True
            for index in range(len(features)):
                if document[index] != 0:
                    if not first_column:
                        mat_file.write(' ')
                    else:
                        first_column = False
                    mat_file.write('%d %f' % (index + 1, document[index]))
            mat_file.write('\n')

    with open(os.path.join('data', '%s.sparse.mat.rclass' % dataset), 'w') as rclass_file:
        for label in labels:
            rclass_file.write(str(label) + '\n')
