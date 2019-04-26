from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Objective functions for document clustering')

    parser.add_argument('--save-dense-matrix', action='store_true')
    parser.add_argument('--objective', type=str, required=True, choices=['I1', 'I2', 'E1', 'H1', 'H2'])
    parser.add_argument('--dataset', type=str, required=True, choices=['reuters8', 'classic4', 'ng20', 'webkb'])

    return parser.parse_args()
