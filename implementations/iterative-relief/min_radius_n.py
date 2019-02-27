import numpy as np
import sklearn.metrics as sk_metrics

def min_radius_n(n, data, target, dist_metric):
    """Compute minimum radius of hypersphere such that for each example in
    the data matrix as the centre the sphere will contain at least n examples from
    same class and n examples from a different class.

    --- Parameters: ---

    n: minimum number of examples from same class and different class a hypersphere with centre in
    each example in the dataset should contain

    data: Matrix containing examples' features as rows

    target: Matrix of target variable values

    dist_metric: distance metric for distance matrix computation

    (see documentation on function pairwise_distances from scikit-learn for 
    valid distance metric specifiers)

    ------

    Returns:
    Minimum acceptable radius of the hypersphere

    Author: Jernej Vivod

    """

    # Allocate array for storing minimum acceptable radius for each example in dataset.
    min_r = np.empty(data.shape[0], dtype=float)

    # Construct distances matrix. Force generation by rows.
    dist_mat = sk_metrics.pairwise_distances_chunked(data, metric=dist_metric, n_jobs=-1, working_memory=0)

    # Go over examples and compute minimum acceptable radius for each example.
    for k in np.arange(data.shape[0]):
        dist_from_e = next(dist_mat)[0]  # Get next row of distances matrix
        msk = target == target[k]        # Get mask for examples from same class.
        dist_same = dist_from_e[msk]     # Get minimum distance that includes n examples from same class.
        dist_diff = dist_from_e[~msk]    # Get minimum distance that includes n examples from different class.
        try:
            min_r[k] = np.max((np.sort(dist_same)[n], np.sort(dist_diff)[n-1]))  # Compute minimum radius for this example.
        except IndexError:
            raise ValueError('Insufficient examples with class {0} for given value of n (n = {1})'.format(target[k], n))

    return np.max(min_r)  # Return maximum of array of minimum acceptable radiuses for each example


# Simple test
if __name__ == '__main__':
    data = np.array([[1, 2, 3], [2, 3, 3], [4, 5, 5], [7, 6, 1]])
    target = np.array([1, 1, 0, 0])
    r = min_radius_n(1, data, target, 'euclidean')