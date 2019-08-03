import numpy as np
import warnings
from algorithms.relieff import Relieff
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin


class ReliefSeq(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the ReliefSeq algorithm

    Brett A. McKinney, Bill C. White, Diane E. Grill, Peter W. Li, Richard B. Kennedy, Gregory A. Poland, Ann L. Oberg.
    ReliefSeq: A Gene-Wise Adaptive-K Nearest-Neighbor Feature Selection Tool 
    for Finding Gene-Gene Interactions and Main Effects in mRNA-Seq Gene Expression Data.

    Author: Jernej Vivod

    """

    def __init__(self, n_features_to_select=10, m=-1, k_max=20,
            dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select.
        self.m = m                                        # sample size of examples for the ReliefF algorithm
        self.k_max = k_max                                # maximal k value
        self.dist_func = dist_func                        # distance function
        self.learned_metric_func = learned_metric_func    # learned metric function


    def fit(self, data, target):

        """
        Rank features using ReliefSeq feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        # Get number of instances with class that has minimum number of instances.
        _, instances_by_class = np.unique(target, return_counts=True)
        min_instances = np.min(instances_by_class)

        # If class with minimal number of examples has less than k examples, issue warning
        # that parameter k was reduced.
        if min_instances < self.k_max:
            warnings.warn("Parameter k_max was reduced to {0} because one of the classes " \
                    "does not have {1} instances associated with it.".format(min_instances, self.k_max), Warning)


        # Run ReliefSeq feature selection algorithm.
        if self.learned_metric_func != None:
            self.rank, self.weights = self._reliefseq(data, target, self.m, min(self.k_max, min_instances), 
                    self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._reliefseq(data, target, self.m, min(self.k_max, min_instances), 
                    self.dist_func, learned_metric_func=None)

        return self


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

        self.fit(data, target)  # Fit training data.
        return self.transform(data)  # Perform feature selection.


    def _reliefseq(self, data, target, m, k_max, dist_func, learned_metric_func):

        """Compute feature scores using ReliefSeq algorithm

        Args:
            data : Array[np.float64] -- matrix containing examples' data as rows
            target : Array[np.int] -- matrix containing the example's target variable value
            m : int -- Sample size to use when evaluating the feature scores
            k_max : int -- k sweep upper limit
            dist_func : Callable[[Array[np.float64], Array[np.float64]], Array[np.float64]] -- function for evaluating
            distances between examples. The function should acept two examples or two matrices of examples and return the dictances.
            **kwargs: can contain argument with key 'learned_metric_func' that maps to a function that accepts a distance
            function and indices of two training examples and returns the distance between the examples in the learned
            metric space.

        Returns:
            Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores

        """

        
        # Initialize matrix of weights by k.
        weights_mat = np.empty((data.shape[1], k_max), dtype=np.float)
        
        # Sweep k from 1 to k_max.
        for k in np.arange(1, k_max+1):

            # Initialize ReliefF algorithm implementation with next value of k.
            clf = Relieff(m=m, k=k, dist_func=dist_func, learned_metric_func=learned_metric_func)

            # Fit data and target.
            clf.fit(data, target)

            # Add weights to matrix.
            weights_mat[:, k-1] = clf.weights
        
        # For each feature choose maximum weight over weights by different values of k.
        weights = np.max(weights_mat, 1)

        # Return feature rankings and weights.
        return rankdata(-weights, method='ordinal'), weights


