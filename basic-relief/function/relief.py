import numpy as np
import pdb
from distance_measures import dist_meas
from functools import partial
from sklearn.datasets import load_iris

def relief(data, target, m, distance_measure):
    """ Get vector of feature quality estimations using the Relief algorithm.

    input: features and class for each training example, parameter m, distance measure
    output: vector of feature quality estimations

    Parameters
    ----------
    data: numpy.ndarray
        data matrix
    target: numpy.ndarray
        target variable values
    m: int
        sample size
    distance_measure: function (list * list -> real list)
        distance function for comparing two examples
    """
    # diff: for a given feature return the difference of values of this feature in the given two examples.
    def diff(f_idx, e1, e2, cont, max_f_val, min_f_val):
        if not cont:  # if feature discrete
            return 0 if e1[f_idx] == e2[f_idx] else 1
        else:         # if feature continuous
            return abs(e1[f_idx] - e2[f_idx])/(max_f_val - min_f_val)

    weights = np.zeros(data.shape[1], dtype=float)  # Set all weights to zero.
    sample_idxs = np.random.choice(np.arange(data.shape[0]), m, replace=False)  # Sample m examples without replacement.

    for idx in sample_idxs:                         # Evaluate features using a sample of m examples.
        e, t = (data[idx, :], target[idx])          # Get sample example data values and target value.
        dist = partial(distance_measure, e)         # Curry distance function with chosen example data vector.
        msk = np.array(list(map(lambda x: True if x == t else False, target)))  # Get mask for examples with same class.

        idx_subset = idx - sum(~msk[:idx+1])        # Get index of sampled example in subset of examples with same class.

        d_same = np.array(list(map(lambda x: dist(x), data[msk, :])))  # Find nearest hit and nearest miss.
        d_same[idx_subset] = np.inf 								   # Set distance of sampled example to itself to infinity.
        d_other = np.array(list(map(lambda x: dist(x), data[~msk, :])))
        nearest_hit = data[msk, :][d_same.argmin(), :]
        nearest_miss = data[~msk, :][d_other.argmin(), :]

        for k in np.arange(data.shape[1]):                                       # Go over features
            max_f_val, min_f_val = np.amax(data[:, k]), np.amin(data[:, k])  # Get maximum and minimum value of current feature across all examples.

            # Update weights
            weights[k] = weights[k] - diff(k, e, nearest_hit, isinstance(e[k], float), max_f_val, min_f_val)/m + \
             diff(k, e, nearest_miss, isinstance(e[k], float), max_f_val, min_f_val)/m

    return weights  # Return vector of feature quality estimates.


# Simple test
manhattan_dist = partial(dist_meas.minkowski_distance, p=1)
iris = load_iris()
w1 = relief(iris.data, iris.target, 100, manhattan_dist)

# Another simple test with tailored data (See Intelligent Systems page 117)
from types import SimpleNamespace
xor = SimpleNamespace()
xor.data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0]])
xor.target = np.array([0, 0, 1, 1, 1, 1, 0, 0])
w2 = relief(xor.data, xor.target, 8, manhattan_dist)


# A test using a dataset of pictures of cats and dogs
import scipy.io as sio
CatDog = sio.loadmat('CatDog.mat')['CatDog']
target = np.hstack((np.repeat(1, 80), np.repeat(0, 80)))
euclidean_dist = partial(dist_meas.minkowski_distance, p=2)
w3 = relief(CatDog, target, CatDog.shape[0], euclidean_dist)

from matplotlib import pyplot as plt
plt.imshow(np.reshape(CatDog[0,:], (64, 64)).T, interpolation='nearest')
plt.show()
