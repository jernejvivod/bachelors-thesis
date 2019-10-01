import numpy as np
import pandas as pd
import scipy.io as sio

from collections import namedtuple, OrderedDict

import os
import sys
import pickle as pkl

from algorithms.relief import Relief
from algorithms.relieff import Relieff
from algorithms.reliefmss import ReliefMSS
from algorithms.reliefseq import ReliefSeq
from algorithms.turf import TuRF
from algorithms.vlsrelief import VLSRelief
from algorithms.iterative_relief import IterativeRelief
from algorithms.irelief import IRelief

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
from sklearn.model_selection import RepeatedKFold

"""
Algorithm evaluations script.

This script produces the scores matrices needed for performing the Bayesian hierarchical correlated t-test.
The results are saved in child 'evaluation_results' folder.

Author: Jernej Vivod

"""

# Set number of CV folds and runs.
NUM_FOLDS_CV = 10
NUM_RUNS_CV = 10
RATIO_FEATURES_TO_SELECT = 0.25

# Set default value for parameter k - number of nearest misses to find (needed for a subset of implemented algorithms).
PARAM_K = 7

# Define named tuple for specifying names of compared algorithms and the scores matrix of comparisons.
alg_vec = namedtuple('alg_vec', 'algorithm scores')

# TODO add others
# algs = OrderedDict([
#     ('iterativerelief', IterativeRelief(max_iter=40)),
#     ('irelief', IRelief(max_iter=40)),
# ])

algs = OrderedDict([
    ('relief', Relief()),
])

# Initialize classifier.
clf = KNeighborsClassifier(n_neighbors=3)

# Set path to datasets folder.
data_dirs_path = os.path.dirname(os.path.realpath(__file__)) + '/datasets/' + 'final9'

# Count datasets and allocate array for results.
num_datasets = len(os.listdir(data_dirs_path))

# Initialize dictionaries for storing results and results counter.
results = dict()
results_count = 0

# Go over all pairs of algorithms (iterate over indices in ordered dictionary).
num_algs = len(algs.keys())
for idx_alg in np.arange(num_algs):

    # Initialize results matrix and results tuple.
    results_mat = np.empty((num_datasets, NUM_FOLDS_CV*NUM_RUNS_CV), dtype=np.float)
    nxt = alg_vec(list(algs.keys())[idx_alg], results_mat)
  
    # Initialize pipelines for evaluating algorithms.
    clf_pipeline = Pipeline([('scaling', StandardScaler()), ('rba', algs[nxt.algorithm]), ('clf', clf)])

    # Initialize row index counter in scores matrix.
    scores_row_idx = 0

    # Go over dataset directories in direstory of datasets.
    for idx_dataset, dirname in enumerate(os.listdir(data_dirs_path)):

        # Load data and target matrices.
        data = sio.loadmat(data_dirs_path + '/' + dirname + '/data.mat')['data']
        target = np.ravel(sio.loadmat(data_dirs_path + '/' + dirname + '/target.mat')['target'])

        # Select num
        num_features_to_select = min(max(2, np.int(np.ceil(RATIO_FEATURES_TO_SELECT*data.shape[1]))), 100)
        clf_pipeline.set_params(rba__n_features_to_select=num_features_to_select)

        print("performing {0} runs of {1}-fold cross validation on dataset '{2}' " \
                "(dataset {3}/{4}).".format(NUM_RUNS_CV, NUM_FOLDS_CV, dirname, idx_dataset+1, num_datasets))
        print("Selecting {0}/{1} features.".format(num_features_to_select, data.shape[1]))

        # Get scores for first algorithm (create pipeline).
        scores_nxt = cross_val_score(clf_pipeline, data, target, 
                cv=RepeatedKFold(n_splits=NUM_FOLDS_CV, n_repeats=NUM_RUNS_CV, random_state=1), verbose=1)

        ## Compute differences of scores.
        #res_nxt = scores1_nxt - scores2_nxt

        # Add row of scores to results matrix.
        nxt.scores[scores_row_idx, :] = scores_nxt

        print("Testing on the '{0}' dataset finished".format(dirname))
       
        # Increment row index counter.
        scores_row_idx += 1

    sio.savemat(nxt.algorithm + '_raw_scores.mat', {'data' : nxt.scores})
    # Save data structure containing results to results dictionary and increment results index counter.
    results[results_count] = nxt
    results_count += 1

