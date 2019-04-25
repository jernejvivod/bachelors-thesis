import numpy as np
from functools import partial
from scipy.spatial.distance import minkowski
import sklearn.metrics as sk_metrics
from scipy.stats import rankdata
import warnings
import pdb

from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=UserWarning)


class IterativeRelief(BaseEstimator, TransformerMixin):

    """TODO"""

    def __init__(self, n_features_to_select=10,  m=100, min_incl=3, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1), max_iter=100, learned_metric_func=None):
        self.m = m
        self.min_incl = min_incl
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.learned_metric_func = learned_metric_func

    def min_radius(self, n, data, target, dist_metric):
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

    
    def fit(self, data, target):
        """
        Rank features using relief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        if self.learned_metric_func != None:
            self.rank, self.weights = self._iterative_relief(data, target, self.m, self.min_incl, self.dist_func, self.max_iter, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._iterative_relief(data, target, self.m, self.min_incl, self.dist_func, self.max_iter)


    def transform(self, data):
        """
        Perform feature selection using computed feature ranks

        Args:
            data : Array[np.float64] -- matrix of examples on which to perform feature selection

        Returns:
            Array[np.float64] -- result of performing feature selection
        """
        # select n_features_to_select best features and return selected features.
        msk = self.rank <= self.n_features_to_select  # Compute mask.
        return data[:, msk]  # Perform feature selection.


    def fit_transform(self, data, target):
        """
        Compute ranks of features and perform feature selection
        Args:
            data : Array[np.float64] -- matrix of examples on which to perform feature selection
            target : Array[np.int] -- vector of target values of examples

        Returns:
            Array[np.float64] -- result of performing feature selection
        """
        self.fit(data, target)  # Fit data
        return self.transform(data)  # Perform feature selection



    def _iterative_relief(self, data, target, m, min_incl, dist_func, max_iter, **kwargs):
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

        min_r = self.min_radius(min_incl, data, target, 'euclidean')  # Get minimum acceptable radius.
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
            idx_sampled = np.random.choice(data.shape[0], data.shape[0] if m == -1 else m)

            # Go over sampled examples.
            for idx in range(10):

                e = data[idx, :]  # Get next sampled example.

                # Compute hypersphere inclusions and distances to examples within the hypersphere.
                same_in_hypsph = np.sum((data[target == target[idx], :] - e)**2, 1) <= min_r**2
                data_same = (data[target == target[idx], :])[same_in_hypsph, :]

                other_in_hypsph = np.sum((data[target != target[idx], :] - e)**2, 1) <= min_r**2
                data_other = (data[target != target[idx]])[other_in_hypsph, :]

                # Compute distances to examples from same class and other classes.
                # Get index of next sampled example in group of examples with same class.
                if 'learned_metric_func' in kwargs:
                    dist = partial(kwargs['learned_metric_func'], lambda x1, x2: dist_func(x1, x2, dist_weights))
                    dist_other = dist(idx, np.where(other_in_hypsph)[0])
                    dist_same = dist(idx, np.where(same_in_hypsph)[0])
                else:
                    dist_other = dist_func(e, data_other, dist_weights)
                    dist_same = dist_func(e, data_same, dist_weights)

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
            dist_weights = feature_weights

            # Check convergence.
            if np.sum(np.abs(feature_weights - feature_weights_prev)) < 0.01:
                convergence = True

            feature_weights_prev = feature_weights

        # Rank features and return rank and distance weights.
        rank = rankdata(-dist_weights, method='ordinal')
        return rank, dist_weights

