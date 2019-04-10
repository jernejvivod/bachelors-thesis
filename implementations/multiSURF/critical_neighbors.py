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
    near_thresh : np.float64 = ex_avg_dist - ex_std  # Get threshold for near neighbors.
    return np.nonzero(dist_mat[ex_idx, msk] < near_thresh)  # Return indices of examples that are considered near neighbors. 
