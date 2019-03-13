import numpy as np
import pdb
from functools import partial
import numba as nb

def relief(data, m, dist_func):
    """Compute feature scores using Relief algorithm

    --- Parameters: ---

    data: Matrix containing examples' data as rows (the last column contains the classes)

    m: Sample size to use when evaluating the feature scores

    dist_func: function for evaluating distances between examples. The function should acept two
    examples or two matrices of examples and return 

    ------

    Returns:
    Array of feature enumerations based on the scores, array of feature scores

    Author: Jernej Vivod

    """

    # update_weights: go over features and update weights.
    @nb.njit
    def _update_weights(data, e, closest_same, closest_other, weights, max_f_vals, min_f_vals):
        for t in np.arange(data.shape[1]-1):
            # Update weights
            weights[t] = weights[t] - (np.abs(e[t] - closest_same[t])/(max_f_vals[t] - min_f_vals[t]))/m + \
                (np.abs(e[t] - closest_other[t])/(max_f_vals[t] - min_f_vals[t]))/m

        return weights  # Return updated weights

    # Initialize all weights to zero.
    weights = np.zeros(data.shape[1]-1, dtype=float)

    # Get maximum and minimum values of each feature
    max_f_vals = np.amax(data, axis=0)
    min_f_vals = np.amin(data, axis=0)

    # Sample m examples without replacement.
    sample_idxs = np.random.choice(np.arange(data.shape[0]), m, replace=False)

    # Evaluate features using a sample of m examples.
    for idx in sample_idxs:
        e = data[idx, :]                                 # Get sample example data.
        msk = np.array(list(map(lambda x: True if x == e[-1] else False, data[:, -1])))  # Get mask for examples with same class.

        # Get index of sampled example in subset of examples with same class.
        idx_subset = idx - sum(~msk[:idx+1])

        # Find nearest hit and nearest miss.
        dist = partial(dist_func, e[:-1])  # Curry distance function with chosen example data vector.
        d_same = dist(data[msk, :-1]) 
        d_same[idx_subset] = np.inf     # Set distance of sampled example to itself to infinity.
        d_other = dist(data[~msk, :-1])
        closest_same = data[msk, :][d_same.argmin(), :]
        closest_other = data[~msk, :][d_other.argmin(), :]

        # ------ weights update ------
        weights = _update_weights(data, e, closest_same, closest_other, weights, max_f_vals, min_f_vals)


    # Create array of feature enumerations based on score.
    rank = np.argsort(weights, 0)[::-1]

    return rank, weights  # Return vector of feature quality estimates.


# Simple test
if __name__ == '__main__':
    def minkowski_distance(e1, e2, p):
        return np.sum(np.abs(e1 - e2)**p, 1)**(1/p)

    test_data = np.loadtxt('rba_test_data2.m')
    rank, weights = relief(test_data, test_data.shape[0], lambda a, b: minkowski_distance(a, b, 2));