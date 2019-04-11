import numpy as np
from scipy.stats import rankdata

def get_pairwise_distances(data, dist_func, mode):
    """
    Compute pairwise distance matrix for examples in training data set.

    Args:
        data : Array[np.float64] -- Matrix of training examples
        dist_func -- function that computes distances between examples
            if mode == 'example' then dist_func : Callable[Array[[np.float64], Array[np.float64l]], np.float64]
            if mode == 'index' then dist_func: Callable[[int, int], np.float64]
        mode : str -- if equal to 'example' the distances are computed in standard metric space by computing
        distances between examples using passed metric function (dist_func). If equal to 'index', the distances
        are computed in learned metric space. In this case, the metric function (dist_func) takes indices of examples
        to compare.

    Returns:
        Pairwise distance matrix : Array[np.float64]

    Raises:
        ValueError : if the mode parameter does not have an allowed value ('example' or 'index')
    """

    if mode == "index":
        # Allocate matrix for distance matrix and compute distances.
        dist_mat = np.empty((data[0].shape, data[0].shape), dtype=np.float64)
        for idx1 in np.arange(data.shape[0]):
            for idx2 in np.arange(idx1, data.shape[0]):
                dist = dist_func(idx1, idx2)
                dist_mat[idx1, idx2] = dist
                dist_mat[idx2, idx1] = dist
        return dist_mat
    elif mode == "example":
       return pairwise_distances(data, metric=dist_func, n_jobs=-1)
    else:
        raise ValueError("Unknown mode specifier")


# TODO do in a vectorized form for all examples.
def critical_neighbours(ex_idx : int, dist_mat : Array[np.gloat64]) -> Array[np.float64]:
    """
    Find neighbours of instance with index inst_idx in radius defined by average distance to other examples and the standard deviation
    of the distances to other examples.

    Args:
        inst_idx : int -- index of the example
        dist_mat : Array[np.float64] -- pairwise distance matrix for examples
    Returns:
        Array[np.int] -- indices of examples that are considered near neighbors of example with index ex_idx.
    """

    msk : Array[int] = np.arange(dist_mat.shape[1]) != ex_idx  # Get mask that excludes ex_idx.
    ex_avg_dist : np.float64 = np.average(dist_mat[ex_idx, msk])  # Get average distance to example with index ex_idx.
    ex_std : np.float64 = np.std(dist_mat[ex_idx, msk]) / 2.0  # Get standard deviation of distances to example with index ex_idx.
    near_thresh : np.float64 = ex_avg_dist - ex_std  # Get threshold for near neighbours.
    return np.nonzero(dist_mat[ex_idx, msk] < near_thresh)  # Return indices of examples that are considered near neighbours. 

# update_weights: go over features and update weights.
@nb.njit
def update_weights(data, e, closest_same, closest_other, weights, weights_mult, max_f_vals, min_f_vals):
    # Update MultiSURF weights as in RELIEFF.
    #
    # Args:
    #   data : Arrax[np.float64] -- training examples
    #   e : Array[np.float64] -- current example
    #   closest_same : Array[np.float64] -- matrix of hits that pass the threshold
    #   closest_other : Array[np.float64] -- matrix of misses that pass the threshold
    #   weights : Array[np.float64] -- feature weights
    #   weights_mult : Array[np.float64] -- probability multiplication for weights.
    #   max_f_vals : Array[np.float64] -- maximum feature values
    #   min_f_vals : Array[np.float64] -- minimum feature values
    #
    # Returns:
    #   Array[np.float64] -- updated feature weights for passed example
    
    # Go over features.
    for t in np.arange(data.shape[1]):

        # Penalty term
        penalty = np.sum(np.abs(e[t] - closest_same[:, t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))

        # Reward term
        reward = np.sum(weights_mult * (np.abs(e[t] - closest_other[:, t])/((max_f_vals[t] - min_f_vals[t] + 1e-10))))

        # Weights update
        weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)

    # Return updated weights.
    return weights


def MultiSURF(data, target, dist_func):

    # TODO, handle learned metrics.
    # Compute pairwise distance matrix.
    dist_mat = get_pairwise_distances(data, dist_func, mode):

    # Get maximum and minimum values of each feature.
    max_f_vals = np.amax(data[:, :], 0)
    min_f_vals = np.amin(data[:, :], 0)

    # Get all unique classes.
    classes = np.unique(target)

    # Get probabilities of classes in training set.
    p_classes = np.vstack(np.unique(target, return_counts=True)).T
    p_classes[:, 1] = p_classes[:, 1]/np.sum(p_classes[:, 1])

    # Initialize feature weights.
    weights = np.zeros(data.shape[1], dtype=float)

    # Compute hits ans misses for each examples (within radius).
    # first row represents the indices of neighbors within threshold and the second row
    # indicates whether an examples is a hit or a miss.
    neighbours_map = dict.fromkeys(np.arange(data.shape[0]))
    for ex_idx in np.arange(data.shape[0]):
        r1 = critical_neighbours(ex_idx, dist_mat)  # Compute indices of neighbours.
        r2 = target[r1] == target[ex_idx]  # Compute whether neighbour hit or miss.
        neighbours_map[ex_idx] = np.vstack((r1, r2)) # Add entry to dictionary.

    # Go over all hits and misses and update weights.
    for ex_idx, neigh_data in neighbours_map.items():
        # TODO compute weights_coefficients
        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 0] != target[ex_idx], 1]
        p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == target[ex_idx], 1])
        weights_mult = np.repeat(p_weights, k)  # Weights multiplier vector

        # Go over all hits and misses (neighbours) and update weights
        for neigh in neigh_data.T:
            weights = update_weights(data, data[ex_idx, :], (data[neigh_data[0], :])[neigh_data[1], :],\
                    (data[neigh_data[0], :])[np.logical_not(neigh_data[1]), :], weights, weights_mult, max_f_vals, min_f_vals)

    # Rank weights and return.
    # Create array of feature enumerations based on score.
    rank = rankdata(-weights, method='ordinal')
    return rank, weights

