import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle as pkl
from collections import namedtuple
import bayesiantests as bt

import matplotlib.pyplot as plt
import seaborn as snb

import scipy.io as sio

import pdb

# Define named tuple that was used to store results.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2 scores')

# Random data for testing ###
acc1 = np.ravel(sio.loadmat('irelief.mat')['res'])         # pleft
acc2 = np.ravel(sio.loadmat('iterative_relief.mat')['res'])     # pright

#acc1 = np.random.rand(10*10)
#acc2 = np.random.rand(10*10)/2

# Names 
names = ("I-RELIEF", "Iterativni Relief")
x = acc2 - acc1

# Set rope values
rope=0.01
pleft, prope, pright = bt.correlated_ttest(x, rope=rope, runs=10, verbose=True, names=names)
with open('results_single_dataset.res', 'a') as f:
    f.write('{0}, {1}, {2}, {3}, {4}\n'.format(names[0], names[1], pleft, prope, pright))


# Plot results.

# generate samples from posterior (it is not necesssary because the posterior is a Student)
samples=bt.correlated_ttest_MC(x, rope=rope, runs=10, nsamples=50000)

# plot posterior
snb.kdeplot(samples, shade=True)

# plot rope region
plt.axvline(x=-rope,color='orange')
plt.axvline(x=rope,color='orange')

# add label
plt.xlabel('Iterativni Relief - I-RELIEF')

plt.show()

