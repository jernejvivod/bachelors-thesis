import numpy as np
import pandas as pd
import scipy.io as sio

from collections import namedtuple, OrderedDict

import os
import sys
import pickle as pkl

from algorithms.relief import Relief
from algorithms.relieff import Relieff

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

"""
Algorithm evaluations script.

This script produces the scores matrices needed for performing the Bayesian hierarchical correlated t-test.
The results are saved in child 'evaluation_results' folder.

Author: Jernej Vivod

"""

# Set number of CV folds and runs.
NUM_FOLDS_CV = 10
NUM_RUNS_CV = 10

# Set default value for parameter k - number of nearest misses to find (needed for a subset of implemented algorithms).
PARAM_K = 10

# Define named tuple for specifying names of compared algorithms and the scores matrix of comparisons.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2 scores')

# Specifiy RBAs to compare (first pair comes from algs1, second from algs2).
GROUP_IDX = 1  # Results index
algs1 = OrderedDict([
    ('Relief', Relief()),
])
algs2 = OrderedDict([
    ('ReliefF', Relieff(k=PARAM_K)),
])

# Initialize classifier.
# clf = KNeighborsClassifier(n_neighbors=3)
clf = SVC(gamma='auto')

# Set path to datasets folder.
data_dirs_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + 'non-noisy'

# Count datasets and allocate array for results.
num_datasets = len(os.listdir(data_dirs_path))

# Initialize dictionaries for storing results and results counter.
results = dict()
results_count = 0

# Go over all pairs of algorithms (iterate over indices in ordered dictionary).
num_algs1 = len(algs1.keys())
num_algs2 = len(algs2.keys())
for idx_alg1 in np.arange(num_algs1):
    for idx_alg2 in np.arange(num_algs2):

        # Initialize results matrix and results tuple.
        results_mat = np.empty((num_datasets, NUM_FOLDS_CV*NUM_RUNS_CV), dtype=np.float)
        nxt = comparePair(list(algs1.keys())[idx_alg1], list(algs2.keys())[idx_alg2], results_mat)
      
        # Initialize pipelines for evaluating algorithms.
        clf_pipeline1 = Pipeline([('scaling', StandardScaler()), ('rba1', algs1[nxt.algorithm1]), ('clf', clf)])
        clf_pipeline2 = Pipeline([('scaling', StandardScaler()), ('rba2', algs2[nxt.algorithm2]), ('clf', clf)])
       
        # Initialize row index counter in scores matrix.
        scores_row_idx = 0

        # Go over dataset directories in direstory of datasets.
        for idx_dataset, dirname in enumerate(os.listdir(data_dirs_path)):

            # Load data and target matrices.
            data = sio.loadmat(data_dirs_path + '/' + dirname + '/data.mat')['data']
            target = np.ravel(sio.loadmat(data_dirs_path + '/' + dirname + '/target.mat')['target'])

            print("performing {0} runs of {1}-fold cross validation on dataset '{2}' " \
                    "(dataset {3}/{4})".format(NUM_RUNS_CV, NUM_FOLDS_CV, dirname, idx_dataset+1, num_datasets))

            # Get scores for first algorithm (create pipeline).
            scores1_nxt = cross_val_score(clf_pipeline1, data, target, cv=RepeatedStratifiedKFold(n_splits=NUM_FOLDS_CV, n_repeats=10, random_state=1), verbose=1)

            # Get scores for second algorithm (create pipeline).
            scores2_nxt = cross_val_score(clf_pipeline2, data, target, cv=RepeatedStratifiedKFold(n_splits=NUM_FOLDS_CV, n_repeats=10, random_state=1), verbose=1)

            # Compute differences of scores.
            res_nxt = scores1_nxt - scores2_nxt

            # Add row of scores to results matrix.
            nxt.scores[scores_row_idx, :] = res_nxt

            print("Testing on the '{0}' dataset finished".format(dirname))
           
            # Increment row index counter.
            scores_row_idx += 1

        # Save data structure containing results to results dictionary and increment results index counter.
        results[results_count] = nxt
        results_count += 1

# Save results to file.
script_path = os.path.abspath(__file__)
script_path = script_path[:script_path.rfind('/')]
with open(script_path + "/evaluation_results/results_group_" + str(GROUP_IDX) + ".p", "wb") as handle:
    pkl.dump(results, handle)

