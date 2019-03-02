import numpy as np
import pdb
from min_radius import min_radius
from scipy.spatial.distance import minkowski

def iterative_relief(data, target, m, min_incl, dist_func, max_iter):
    """Compute feature ranks and scores using Iterative Relief algorithm

    --- Parameters: ---

    data: Matrix containing examples' features in rows

    target: Matrix containing the target variable values

    m: Sample size to use when evaluating the feature scores

    min_incl: the minimum number of examples from same and other 
    classes that a hypersphere centered at each examples should contain.

    dist_func: distance function for evaluating distance between examples. 
    The function should be able to take two matrices of examples and return a vector of distances
    between the examples. The distance function should accept a weights parameter.

    max_iter: Maximum number of iterations to compute

    ------

    Returns:
    Array of feature enumerations based on the scores, array of feature scores

    Author: Jernej Vivod

    """

    min_r = min_radius(min_incl, data, target, 'euclidean')  # Get minimum acceptable radius.
    dist_weights = np.ones(data.shape[1], dtype=float)       # Initialize distance weights.  

    # Initialize iteration counter, convergence indicator and
    # Array for storing feature weights from previous iteration.
    iter_count = 0
    convergence = False
    feature_weights_prev = np.zeros(data.shape[1], dtype=float)

    # Iterate
    while iter_count < max_iter and not convergence:
        iter_count += 1   # Increment iteration counter.

        # Reset feature weights to zero and sample examples.
        feature_weights = np.zeros(data.shape[1], dtype=float)
        idx_sampled = np.random.choice(data.shape[0], m)

        # Go over sampled examples.
        for idx in range(10):

            e = data[idx, :]  # Get next sampled example.

            # Compute hypersphere inclusions and distances to examples within the hypersphere.
            same_in_hypsph = np.sum((data[target == target[idx], :] - e)**2, 1) <= min_r**2
            data_same = (data[target == target[idx], :])[same_in_hypsph, :]
            dist_same = dist_func(e, data_same, dist_weights)

            other_in_hypsph = np.sum((data[target != target[idx], :] - e)**2, 1) <= min_r**2
            data_other = (data[target != target[idx]])[other_in_hypsph, :]
            dist_other = dist_func(e, data_other, dist_weights)


            # *********** Feature Weights Update ***********
            w_miss = np.maximum(0, 1 - (dist_other**2/min_r**2))   # TODO compare zero to every column value
            w_hit = np.maximum(0, 1 - (dist_same**2/min_r**2))

            numerator1 = np.sum(np.abs(e - data_other) * w_miss[np.newaxis].T, 0)
            denominator1 = np.sum(w_miss) + np.finfo(float).eps

            numerator2 = np.sum(np.abs(e - data_same) * w_hit[np.newaxis].T, 0)
            denominator2 = np.sum(w_hit) - 1 + np.finfo(float).eps

            feature_weights += numerator1/denominator1 - numerator2/denominator2
            # **********************************************

        # Update distance weights by feature weights.
        dist_weights += feature_weights

        # Check convergence.
        if np.sum(np.abs(feature_weights - feature_weights_prev)) < 0.01:
            convergence = True

        feature_weights_prev = feature_weights

    # Rank features and return rank and distance weights.
    rank = np.argsort(dist_weights, 0)[::-1]
    return rank, dist_weights


# Test
if __name__ == '__main__':

    def minkowski_distance_w(e1, e2, w, p):
        return np.sum(np.abs(w*(e1 - e2))**p, 1)**(1/p)

    test_data = np.loadtxt('rba_test_data2.m')

    rank, weights = iterative_relief(test_data[:, :-1], test_data[:, -1], test_data.shape[0], 1, lambda a, b, w: minkowski_distance_w(a, b, w, 2), 100);