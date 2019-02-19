import numpy as np
import gzip


def get_tidy_data(filename):
    with gzip.open(filename, 'rt') as csvfile:
        raw_X = np.genfromtxt(csvfile, delimiter=',', encoding='utf-8', dtype='str')
        features = np.hstack((np.matrix(raw_X[1:, [0]]), raw_X[1:, 2:8]))
        target = raw_X[1:, [1]]
        return features, target
