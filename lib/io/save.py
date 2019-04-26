import os


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
