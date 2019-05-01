import numpy as np
from scipy.stats import rankdata

def rand_sel(data, **kwargs):
    """
    Assign random weights to features and rank them

    Args:
        data : Array[np.float64] -- training examples
        kwargs -- ignored

    Returns:
        random ranking of features and randomly assigned feature weights
    """
    weights = np.random.rand(data.shape[1])
    rank = rankdata(-weights, method='ordinal')
    return rank, weights
