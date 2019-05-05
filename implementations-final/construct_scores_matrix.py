import numpy as np
import pandas as pd
import scipy.io as sio

from collections import namedtuple

import os
import sys

import pdb

# Define named tuple for specifying names of compared algorithms.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2')




##### COMPARED ALGORITHMS PAIR #####

# Specify pair of algorithms to compare.
compared_pair = comparePair('relieff', 'iterative-relief')

####################################

# Specify whether to use noisy (added noise features) or non-noisy (no added noise features) datasets.
NOISY = True




# Specifiy RBAs. TODO
algs = {
    'relief' : None,
    'relieff' : None,
    'iterative-relief' : None,
    'irelief' : None
}

results_mat = None # TODO

# Specify path to dataset directories.
data_dirs_path = sys.path[0] + '/datasets/' + 'noisy' if NOISY else 'non-noisy'
# Go over dataset directories.
for dirname in os.listdir(data_dirs_path):

    # Load data and target matrices.
    data = sio.loadmat(data_dirs_path + '/' + dirname + '/data.mat')['data']
    target = np.ravel(sio.loadmat(data_dirs_path + '/' + dirname + '/target.mat')['target'])

    # Get scores for first algorithm

    # Get scores for second algorithm

    # Create next row in results matrix and add.
