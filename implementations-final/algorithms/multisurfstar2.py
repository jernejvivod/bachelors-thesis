import numpy as np
import os
from scipy.stats import rankdata
from functools import partial
from nptyping import Array
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, TransformerMixin
from julia import Julia
jl = Julia(compiled_modules=False)


class MultiSURFStar(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the MultiSURFStar algorithm

    Granizo-Mackenzie Delaney; Moore, Jason H.
    Multiple Threshold Spatially Uniform ReliefF for the Genetic Analysis of Complex Human Diseases. 

    author: Jernej Vivod

    """

    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # Number of features to select.
        self.dist_func = dist_func                        # Distance function to use.
        self.learned_metric_func = learned_metric_func    # learned metric function.

        # Use function written in Julia programming language to update feature weights.
        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_multisurfstar2.jl")


    def fit(self, data, target):
        """
        Rank features using MultiSURFStar feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """
        
        # Fit training data.
        if self.learned_metric_func != None:
            self.rank, self.weights = self._multisurfstar(data, target, self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._multisurfstar(data, target, self.dist_func)

        return self


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
            Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores

        """

        # Initialize weights.
        weights = np.zeros(data.shape[1], dtype=np.float)
        
        # Get maximum and minimum feature values.
        max_f_vals = np.max(data, 0)
        min_f_vals = np.min(data, 0)

        for idx in np.arange(data.shape[0]):

            # Get next example
            e = data[idx, :]
            target_e = target[idx]

            # Compute distances from current examples to all other examples.
            if 'learned_metric_func' in kwargs:
                dist_func_learned = partial(kwargs['learned_metric_func'], dist_func)
                dists = np.array([dist_func_learned(idx, idx_other) for idx_other in np.arange(data.shape[0])])
            else:
                dists = dist_func(data[idx, :], data)


            # Compute mean and standard deviation of distances and set thresholds.
            t_nxt = np.mean(dists[np.arange(data.shape[0]) != idx])
            sigma_nxt = np.std(dists[np.arange(data.shape[0]) != idx])
            thresh_near = t_nxt - sigma_nxt/2.0
            thresh_far = t_nxt + sigma_nxt/2.0


            # Get mask of examples that are close.
            msk_near = dists < thresh_near
            msk_near[idx] = False

            # Get mask of examples that are far.
            msk_far = dists > thresh_far
            
            # Get target values for considered regions.
            target_close = target[msk_near]
            target_far = target[msk_far]
            
            # Get class values of miss neighbours.
            classes_other_near = target[np.logical_and(msk_near, target != target[idx])]
            classes_other_far = target[np.logical_and(msk_far, target != target[idx])]
            
            # Get masks for considered regions.
            hit_neigh_mask_near = np.logical_and(msk_near, target == target[idx])
            hit_neigh_mask_far = np.logical_and(msk_far, target == target[idx])
            miss_neigh_mask_near = np.logical_and(msk_near, target != target[idx])
            miss_neigh_mask_far = np.logical_and(msk_far, target != target[idx])

            
            # Compute probability weights for misses in considered regions.            
            weights_mult1 = np.empty(classes_other_near.size, dtype=np.float)
            u, c = np.unique(classes_other_near, return_counts=True)
            neighbour_weights = c/classes_other_near.size
            for i, val in enumerate(u):
                weights_mult1[np.where(classes_other_near == val)] = neighbour_weights[i]

            weights_mult2 = np.empty(classes_other_far.size, dtype=np.float)
            u, c = np.unique(classes_other_far, return_counts=True)
            neighbour_weights = c/classes_other_far.size
            for i, val in enumerate(u):
                weights_mult2[np.where(classes_other_far == val)] = neighbour_weights[i]


            ### WEIGHTS UPDATE ###

            # Update feature weights for near examples.
            weights_near = self._update_weights(data, e[np.newaxis], data[hit_neigh_mask_near, :], 
                    data[miss_neigh_mask_near, :], weights[np.newaxis], weights_mult1[np.newaxis].T,
                    max_f_vals[np.newaxis], min_f_vals[np.newaxis])

            # Update feature weights for far examples.
            weights_far = self._update_weights(data, e[np.newaxis], data[hit_neigh_mask_far, :], 
                    data[miss_neigh_mask_far, :], weights[np.newaxis], weights_mult2[np.newaxis].T,
                    max_f_vals[np.newaxis], min_f_vals[np.newaxis])
            
            # Subtract scoring for far examples. Subtract previous value of weights to get delta.
            weights = weights_near - (weights_far - weights)

            ### /WEIGHTS UPDATE ###



        # Create array of feature enumerations based on score.
        rank = rankdata(-weights, method='ordinal')
        return rank, weights


