from lib.args import get_args
from lib.io import load, save
from lib import objectives

if __name__ == '__main__':
    # Get commandline args from args.py
    args = get_args()

    if args.dataset == 'reuters8':
        features, labels = load.reuters8()
    elif args.dataset == 'classic4':
        features, labels = load.classic4()
    elif args.dataset == 'ng20':
        features, labels = load.ng20()
    elif args.dataset == 'webkb':
        features, labels = load.webkb()
    else:
        raise Exception('Unknown dataset')

    if args.save_dense_matrix:
        save.dense_matrix(features, labels, args.dataset)
    if args.save_sparse_matrix:
        save.sparse_matrix(features, labels, args.dataset)

    if args.objective == 'I1':
        objective_value = objectives.I1(features, labels)
    elif args.objective == 'I2':
        objective_value = objectives.I2(features, labels)
    elif args.objective == 'E1':
        objective_value = objectives.E1(features, labels)
    elif args.objective == 'H1':
        objective_value = objectives.H1(features, labels)
    elif args.objective == 'H2':
        objective_value = objectives.H2(features, labels)
    else:
        raise Exception('Unknown objective')

    print('%s objective value for %s ground-truth clustering: %f' % (args.objective, args.dataset, objective_value))
