from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from typing import Callable
from nptyping import Array

# Idea - reduce the dimensionality of examples and measure distances between examples in this space.

def get_dist_func(data : Array[np.float64], target : int, n : int,  metric : Callable[[np.float64, np.float64], np.float64]) -> Callable[[int, int], np.float64]:
    """
    Get function that returns distances between examples in learned space.

    Args:
        data : Array[np.float64] - training data_trans
        target : int - target variable values (classes of training examples)
        n : int - number of components to keep
        dist_func : Callable[[np.float64, np.float64], np.float64] - metric for measuring distances in learned space.
    Returns:
        Callable[[np.float64, np.float64], np.float64] - function that takes indices of training examples
                                                and returns the distance between them in learned metric space using
                                                specified metric.
    """

    # Get transformed data.
    data_trans : Array[np.float64] = LinearDiscriminantAnalysis(n_components=n).fit_transform(StandardScaler().fit_transform(data), target)

    # Computing distance:
    def dist_func_res(i1 : int, i2 : int) -> np.float64:
        """ 
        distance function that takes indices of examples in training set and returns distance
        in learned space using specified distance metric.

        Args:
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
    target : Array[np.int] = load_iris()['target']  # Get classes of examples from Iris dataset.
    dist_func : Callable[[np.float64, np.float64], np.float64] = \
            get_dist_func(data, target, 3, lambda x1, x2: (np.sum(np.abs(x1 - x2)**2))**(1/2))  # Get distance function. Use euclidean distance as metric in
                                                                                                # learned space.
    print("distance between first and second example in learned metric space: {0}".format(dist_func(1, 2)));

