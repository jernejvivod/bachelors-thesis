from sklearn.decomposition import PCA
import numpy as np
from nptyping import Array

# Idea - reduce the dimensionality of examples and measure distances between examples in this space.

get_dist_func(data : Array[np.float64], dist_func : Callable[[np.float64, np.float64], np.float64]) -> Callable[[np.float64, np.float64], np.float64]:
    """ TODO DOC: implemented using a closure.  """

    # TODO NORMALIZE DATA

    # Get transformed data.
    data_trans : Array[np.float64] = PCA(n_components='mle').fit_transform(data)

    # Computing distance:
    def dist_func_res(i1 : int, i2 : int) -> np.float64:
        """ TODO DOC: function takes indices of examples  """

        # Compute distances from examples in PCA space using specified distance function.
        return dist_func(data_trans[i1, :], data_trans[i2, :])

    return dist_func_res

