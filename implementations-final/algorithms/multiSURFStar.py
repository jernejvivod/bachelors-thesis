import numpy as np
from scipy.stats import rankdata
from functools import partial
from nptyping import Array
from sklearn.metrics import pairwise_distances

from sklearn.base import BaseEstimator, TransformerMixin

class MultiSURFStar(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the MultiSURFStar algorithm

        author: Jernej Vivod

    """

    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2: np.sum(np.logical_xor(x1, x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # Number of features to select.
        self.dist_func = dist_func                        # Distance function to use.
        self.learned_metric_func = learned_metric_func    # learned metric function.

    def fit(self, data, target):
        """
        Rank features using MultiSURFStar feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        if self.learned_metric_func != None:
            self.rank, self.weights = self._multiSURFStar(data, target, self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._multiSURFStar(data, target, self.dist_func)


    def transform(self, data):
        """
        Perform feature selection using computed feature ranks

        Args:
            data : Array[np.float64] -- matrix of examples on which to perform feature selection

        Returns:
            Array[np.float64] -- result of performing feature selection
        """

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

        self.fit(data, target)
        return self.transform(data)

    def _get_pairwise_distances(self, data, dist_func, mode):
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

        # If computing distances between examples by referencing them by indices.
        if mode == "index":
            # Allocate matrix for distance matrix and compute distances.
            dist_func_adapter = lambda x1, x2 : dist_func(int(np.where(np.sum(np.equal(x1, data), 1) == data.shape[1])[0][0]),
                    int(np.where(np.sum(np.equal(x2, data), 1) == data.shape[1])[0][0]))
            return pairwise_distances(data, metric=dist_func_adapter)
        elif mode == "example":  # Else if passing in examples.
            return pairwise_distances(data, metric=dist_func)
        else:
            raise ValueError("Unknown mode specifier")


    def _multisurfstar(self, data, target, dist_func, **kwargs):

        """Compute feature scores using multiSURFStar algorithm

        Args:
            data : Array[np.float64] -- Matrix containing examples' data as rows
            target : Array[np.int] -- matrix containing the example's target variable value
            dist_func : Callable[[Array[np.float64], Array[np.float64]], Array[np.float64]] -- function for evaluating
            distances between examples. The function should acept two examples or two matrices of examples and return the dictances.
            **kwargs: can contain argument with key 'learned_metric_func' that maps to a function that accepts a distance
            function and indices of two training examples and returns the distance between the examples in the learned
            metric space.

        Returns:
            Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores

        """

        # Compute pairwise distances.
        if 'learned_metric_func' in kwargs:
            dist_func_learned = partial(kwargs['learned_metric_func'], dist_func)
            pairwise_dist = self._get_pairwise_distances(data, dist_func_learned, mode="index")
        else:
            pairwise_dist = self._get_pairwise_distances(data, dist_func, mode="example")

        # Initialize weights.
        weights = np.zeros(data.shape[1], dtype=np.float)

        for row_idx, dist_mat_row in enumerate(pairwise_dist[:-1, :]):

            # Get next example
            e = data[row_idx, :]
            target_e = target[row_idx]

            # Get mask of examples that are close.
            msk_close = (dist_mat_row < thresh_near)[row_idx+1:]
            # Get mask of examples that are far.
            msk_far = (dist_mat_row > thresh_far)[row_idx+1:]

            # Get examples that are close.
            examples_close = data[msk_close, :]
            target_close = target[msk_close]
            
            # Get examples that are far.
            examples_far = data[msk_fat, :]
            target_far = target[msk_far]

            # Get considered features of close examples.
            features_close = e != examples_close
            # Get considered features of far examples.
            features_far = e == examples_far 

            # Get mask for close examples with same class.
            msk_same_close = target_close == target_e

            # Get mask for far examples with same class.
            msk_same_far = target_far == target_e


            ### WEIGHTS UPDATE ###

            # Get penalty weights update values for close examples. 
            wu_close_penalty = np.sum(features_close[msk_same_close, :], 0)
            # Get reward weights update values for close examples. 
            wu_close_reward = np.sum(features_close[np.logical_not(msk_same_close), :], 0)

            # Get penalty weights update values for far examples.
            wu_far_penalty = np.sum(features_far[msk_same_far, :], 0)
            # Get reward weights update values for close examples. 
            wu_far_reward = np.sum(features_far[np.logical_not(msk_same_far), :], 0)

            # Update weights.
            weights = weights - (wu_close_penalty + wu_far_penalty) + (wu_close_reward + wu_close_reward)

            ### /WEIGHTS UPDATE ###

        # Create array of feature enumerations based on score.
        rank = rankdata(-weights, method='ordinal')
        return rank, weights

