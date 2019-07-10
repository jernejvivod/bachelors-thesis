from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Callable
from nptyping import Array

# Idea - reduce the dimensionality of examples and measure distances between examples in this space.

def get_dist_func(data : Array[np.float64]) -> Callable[[Callable[[np.float64, np.float64], np.float64], np.int, np.int], np.float64]:
    """
    Get function that returns distances between examples in learned space.

    Args:
        data : Array[np.float64] - training data_trans
    Returns:
        Callable[[Callable[[np.float64, np.float64], np.float64], np.float64, np.float64], np.float64] - 
        function that takes indices of training examples and returns the distance between them in learned 
        metric space using specified metric.
    """

    # Get transformed data.
    data_trans : Array[np.float64] = PCA(n_components='mle').fit_transform(StandardScaler().fit_transform(data))

    # distance_function definition
    def dist_func_res(metric : Callable[[np.float64, np.float64], np.float64], i1 : np.int, i2 : np.int) -> np.float64:
        """ 
        distance function that takes metric function and indices of examples in training set and returns distance
        in learned space using specified distance metric.

        Args:
            metric: Callable[[np.flaot64, np.float64], np.float64] - metric to use in learned metric space.
            i1 : int - index of first training example
            i2 : int - index of second training example
        Returns:
            np.float64 - distance in learned metric space using specified metric
                    between specified training examples.
        """

        # Compute distance in learned metric space using specified metric.
        return metric(data_trans[i1, :], data_trans[i2, :])

    return dist_func_res  # Return distance function.

if __name__ == '__main__':
    from sklearn.datasets import load_iris  # Import function that loads the Iris dataset.
    data : Array[np.float64] = load_iris()['data']  # Get examples from Iris dataset.
    dist_func : Callable[[Callable[[np.float64, np.float64], np.float64], np.float64, np.float64], np.float64] = get_dist_func(data)  # Get distance function. 
    print("distances: {0}".format(dist_func(lambda x1, x2 : (np.sum(np.abs(x1 - x2)**2, 1))**(1/2), 1, [0, 1, 2])));
