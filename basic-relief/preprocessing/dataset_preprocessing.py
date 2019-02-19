import numpy as np
import gzip
import pdb


def get_tidy_data(filename):
    with gzip.open(filename, 'rt') as csvfile:
        raw_X = np.genfromtxt(csvfile, delimiter=',', encoding='utf-8', dtype='str')
        features = np.hstack((raw_X[1:, [0]], raw_X[1:, 2:8]))
        target = raw_X[1:, 1]
        for k, col in enumerate(features.T):
            col_temp = np.array(list(map(lambda x: '0' if x == '?' else x, col)))
            avg_val = sum(col_temp.astype(float))/len(col_temp)
            for i, val in enumerate(features[:, k]):
                if val == '?':
                    features[i, k] = str(avg_val)
        return features.astype(float), target.astype(int)