import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from PIL import Image

from algorithms.relieff import Relieff
from algorithms.relief import Relief
from algorithms.swrfStar import SWRFStar

# number of best features to mark.
N_TO_SELECT = 500

# Load the CatDog dataset and create a target vector where: 0 - cat, 1 - dog
data = sio.loadmat('./datasets/CatDog.mat')['CatDog']
target = np.hstack((np.repeat(0, 80), np.repeat(1, 80)))

# Get mean cat and dog images.
mean_cat = np.mean(data[:60], 0)
mean_dog = np.mean(data[60:], 0)

# Create dictionary of initialized RBAs.
algs = {'Relief' : Relief(), 'ReliefF' : Relieff(k=10), 'SWRFStar' : SWRFStar()}

# Go over RBAs.
for alg_name in algs.keys():
    alg = algs[alg_name]
    alg.fit(data, target)

    mean_cat_nxt = mean_cat.copy()
    mean_dog_nxt = mean_dog.copy()
    mean_cat_nxt_s = np.dstack((mean_cat_nxt, mean_cat_nxt, mean_cat_nxt))
    mean_dog_nxt_s = np.dstack((mean_dog_nxt, mean_dog_nxt, mean_dog_nxt))
   
    # Mark selected best features.
    mean_cat_nxt_s[0, alg.rank < N_TO_SELECT, 0] = 255
    mean_dog_nxt_s[0, alg.rank < N_TO_SELECT, 0] = 255

    import pdb
    pdb.set_trace()

    # Save images.
    plt.imsave("./fs-visualization-catdog/" + alg_name + "_cat", mean_cat_nxt_s.reshape(64, 64, 3).astype(np.ubyte).transpose(1, 0, 2), cmap='gray')
    plt.imsave("./fs-visualization-catdog/" + alg_name + "_dog", mean_dog_nxt_s.reshape(64, 64, 3).astype(np.ubyte).transpose(1, 0, 2), cmap='gray')
