import numpy as np
import numpy as np
from functools import partial
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from algorithms.relieff import Relieff


class VLSRelief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the vlsRelief algorithm

    Margaret J. Eppstein, Paul Haake.
    Very large scale ReliefF for genome-wide association analysis.
    
    Author: Jernej Vivod
    """

    def __init__(self, n_features_to_select=10, num_partitions_to_select=10, 
            num_subsets=10, partition_size=5, m=-1, k=5, 
            dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.num_partitions_to_select = num_partitions_to_select  # number of partitions to combine to form subset of features
        self.num_subsets = num_subsets  # number of subsets to evaluate
        self.partition_size = partition_size  # size of partitions
        self.m = m                            # the m parameter for ReliefF algorithm (example sample size)
        self.k = k                            # the k parameter for ReliefF algorithm (number of closest examples from each class to consider)
        self.dist_func = dist_func            # distance function to use
        self.learned_metric_func = learned_metric_func  # learned metric function


    def fit(self, data, target):
        """
        Rank features using vlsRelief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        # Run VLSRelief feature selection algorithm.
        if self.learned_metric_func != None:
            self.rank, self.weights = self._vlsrelief(data, target, self.num_partitions_to_select, 
                    self.num_subsets, self.partition_size, self.m, self.k, self.dist_func, 
                    learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._vlsrelief(data, target, self.num_partitions_to_select, 
                    self.num_subsets, self.partition_size, self.m, self.k, self.dist_func)
        return self


    def transform(self, data, target):
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


    def _vlsrelief(self, data, target, num_partitions_to_select, num_subsets, 
            partition_size, m, k, dist_func, **kwargs):

        """Compute feature scores and ranking using vlsRelief algorithm

        Args:
            data : Array[np.float64] -- matrix containing examples' data as rows
            target : Array[np.int] -- matrix containing the example's target variable value
            num_partitions_to_select : int -- number of partitions to use to construct feature subset
            num_subsets : int -- number of subsets to evaluate for local weights
            partition_size : int -- size of feature partitions
            m : int -- Sample size to use when evaluating the feature scores
            k : int -- Number of closest examples from each class to use
            dist_func : Callable[[Array[np.float64], Array[np.float64]], Array[np.float64]] -- function for evaluating
            distances between examples. The function should acept two examples or two matrices of examples and return the dictances.
            **kwargs: can contain argument with key 'learned_metric_func' that maps to a function that accepts a distance
            function and indices of two training examples and returns the distance between the examples in the learned
            metric space.

        Returns:
            Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores

        """

        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.float)

        # Get array of feature indices.
        feat_ind = np.arange(data.shape[1])

        # Get indices of partition starting indices.
        feat_ind_start_pos = np.arange(0, data.shape[1], partition_size)

        # Initialize ReliefF algorithm.
        if 'learned_metric_func' in kwargs:
            relieff = Relieff(n_features_to_select=self.n_features_to_select, 
                    m=m, k=k, dist_func=dist_func, learned_metric_func=learned_metric_func)
        else:
            relieff = Relieff(n_features_to_select=self.n_features_to_select, 
                    m=m, k=k, dist_func=dist_func)

        # Go over subsets and compute local ReliefF scores.
        for i in np.arange(num_subsets):

            # Randomly select k partitions and combine them to form a subset of features of examples.
            ind_sel = np.ravel([np.arange(el, el+partition_size) for el in np.random.choice(feat_ind_start_pos, num_partitions_to_select)])
            ind_sel = ind_sel[ind_sel <= feat_ind[-1]]
            
            # Perform ReliefF algorithm on subset to obtain local weights.
            relieff = relieff.fit(data[:, ind_sel], target)

            # Update weights using local weights.
            weights[ind_sel] = np.maximum(weights[ind_sel], relieff.weights)

        
        # Return feature rankings and weights.
        return rankdata(-weights, method='ordinal'), weights
 
