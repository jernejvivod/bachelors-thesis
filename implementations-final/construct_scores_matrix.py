import numpy as np
import pandas as pd
import scipy.io as sio

from collections import namedtuple, OrderedDict

import os
import sys

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
from sklearn.model_selection import KFold

"""
TODO: compare algorithms augmented with turf.
"""


NUM_FOLDS_CV = 2
NUM_RUNS_CV = 1

NOISY = False

# Define named tuple for specifying names of compared algorithms and the scores matrix of comparisons.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2 scores')


# Specifiy RBAs to compare.
algs = OrderedDict([
    ('Relief', Relief()),
    ('ReliefF', Relieff(k=3))
])

# Initialize classifier.
# clf = KNeighborsClassifier(n_neighbors=3)
clf = SVC(gamma='auto')


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
        nxt = comparePair(list(algs.keys())[idx_alg_1], list(algs.keys())[idx_alg_2], results_mat)

        print("Comparing {0} and {1}".format(nxt.algorithm1, nxt.algorithm2))
       
        # Initialize pipelines for evaluating algorithms.
        clf_pipeline1 = Pipeline([('scaling', StandardScaler()), ('rba1', algs[nxt.algorithm1]), ('clf', clf)])
        clf_pipeline2 = Pipeline([('scaling', StandardScaler()), ('rba2', algs[nxt.algorithm2]), ('clf', clf)])
        
        # Initialize row index counter in scores matrix.
        row_idx = 0

        # Go over dataset directories.
        for dirname in os.listdir(data_dirs_path):

            print("{0} ({1})".format(dirname, row_idx))
            
            # Load data and target matrices.
            data = sio.loadmat(data_dirs_path + '/' + dirname + '/data.mat')['data']
            target = np.ravel(sio.loadmat(data_dirs_path + '/' + dirname + '/target.mat')['target'])
            
            # Perform 10 runs of 10-fold cross validation.
            for idx_run in np.arange(NUM_RUNS_CV):

                print("idx_run == {0}".format(idx_run))

                # Get scores for first algorithm (create pipeline).
                scores1_nxt = cross_val_score(clf_pipeline1, data, target, cv=KFold(NUM_FOLDS_CV, shuffle=True))

                # Get scores for second algorithm (create pipeline).
                scores2_nxt = cross_val_score(clf_pipeline2, data, target, cv=KFold(NUM_FOLDS_CV, shuffle=True))

                # Compute differences of scores.
                res_nxt = scores1_nxt - scores2_nxt

                # Create next row in results matrix and add.
                nxt.scores[row_idx, idx_run*NUM_RUNS_CV:idx_run*NUM_RUNS_CV + NUM_FOLDS_CV] = res_nxt
            
            row_idx += 1

        import pdb
        pdb.set_trace()

        results[results_count] = nxt
        results_count += 1
