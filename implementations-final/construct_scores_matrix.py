import numpy as np
import pandas as pd
import scipy.io as sio

from collections import namedtuple, OrderedDict

import os
import sys


"""
TODO: compare algorithms augmented with turf.
"""


NUM_FOLDS_CV = 10
NUM_RUNS_CV = 10

NOISY = False

# Define named tuple for specifying names of compared algorithms and the scores matrix of comparisons.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2 scores')


# Specifiy RBAs to compare.
algs = OrderedDict([
    ('BoostedSURF', None),
    ('I-relief', None),
    ('Iterative-Relief', None),
    ('MultiSURF', None),
    ('MultiSURFStar', None),
    ('Relief', None),
    ('ReliefF', None),
    ('ReliefMMS', None),
    ('ReliefSeq', None),
    ('SURF', None),
    ('SURFStar', None),
    ('SwrfStar', None),
    ('TuRF', None),
    ('VLSReliefF', None),
    ('Evaporative Cooling Relief', None),
    ('Adaptive Relief', None),
    ('STIR', None),
])


# Specify path to dataset directories (noisy data or non-noisy data).
if NOISY:
    data_dirs_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + 'noisy'
else:
    data_dirs_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + 'non-noisy'


# Count datasets and allocate array for results.
num_datasets = len(os.listdir(data_dirs_path))

# Initialize dictionaries for storing results and results counter.
results = dict()
results_count = 0

# Go over all pairs of algorithms (iterate over indices in ordered dictionary).
num_algs = len(algs.keys())
for idx_alg_1 in np.arange(num_algs-1):
    for idx_alg_2 in np.arange(idx_alg_1+1, num_algs):

        # Initialize results matrix and results tuple.
        results_mat = np.empty((num_datasets, NUM_FOLDS_CV*NUM_RUNS_CV), dtype=np.float)
        nxt = comparePair(list(algs.keys())[idx_alg_1], list(algs[idx_alg_2])[idx_alg_2], results_mat)

        # Go over dataset directories.
        for dirname in os.listdir(data_dirs_path):

            # Load data and target matrices.
            data = sio.loadmat(data_dirs_path + '/' + dirname + '/data.mat')['data']
            target = np.ravel(sio.loadmat(data_dirs_path + '/' + dirname + '/target.mat')['target'])

            # Get scores for first algorithm.

            # Get scores for second algorithm.

            # Compute differences of scores.

            # Create next row in results matrix and add.
            nxt.scores[row_idx, :] = res_nxt

        results[results_count] = nxt
        results_count += 1
