import numpy as np
from scipy.stats import rankdata
from functools import partial
from sklearn.metrics import pairwise_distances
import os
from sklearn.base import BaseEstimator, TransformerMixin
from julia import Julia
jl = Julia(compiled_modules=False)

class SURFStar(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the SURFStar algorithm

    Casey S. GreeneDaniel S. HimmelsteinJeff KiralisJason H. Moore.
    The Informative Extremes: Using Both Nearest and Farthest Individuals Can 
    Improve Relief Algorithms in the Domain of Human Genetics.

    Author: Jernej Vivod

    """

    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2)), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.dist_func = dist_func                        # metric function
        self.learned_metric_func = learned_metric_func    # learned metric function

        # Use function written in Julia programming language to update feature weights.
        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_surfstar3.jl")


    def fit(self, data, target):

        """
        Rank features using SURFStar feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        # Run SURFStar feature selection algorithm.
        if self.learned_metric_func != None: 
            self.rank, self.weights = self._surfStar(data, target, self.dist_func, 
                    learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._surfStar(data, target, self.dist_func)

        return self


    def transform(self, data):

        """
        Perform feature selection using computed feature ranks.

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
            dist_func_adapter = lambda x1, x2 : dist_func(np.int(np.where(np.sum(np.equal(x1, data), 1) == data.shape[1])[0][0]),
                    np.int(np.where(np.sum(np.equal(x2, data), 1) == data.shape[1])[0][0]))
            return pairwise_distances(data, metric=dist_func_adapter)
        elif mode == "example":  # Else if passing in examples.
            return pairwise_distances(data, metric=dist_func)
        else:
            raise ValueError("Unknown mode specifier")


    def _surfStar(self, data, target, dist_func, **kwargs):

        """Compute feature scores using SURFStar algorithm

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

        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.float)

        # Compute weighted pairwise distances.
        if 'learned_metric_func' in kwargs:
            dist_func_learned = partial(kwargs['learned_metric_func'], dist_func)
            pairwise_dist = self._get_pairwise_distances(data, dist_func_learned, mode="index")
        else:
            # Get weighted distance function.
            pairwise_dist = self._get_pairwise_distances(data, dist_func, mode="example")

        # Get mean distance between all examples.
        mean_dist = np.float(np.sum(pairwise_dist))/np.float(np.size(pairwise_dist))
        
        # Compute maximal and minimal feature values.
        max_f_vals = np.max(data, 0)
        min_f_vals = np.min(data, 0)

        # Go over examples.
        for idx in np.arange(data.shape[0]):

           
            # Select next example.
            e = data[idx, :]

            ### NEIGHBOUR INDICES ###

            # Get indices of near neighbours.
            neigh_mask_near = pairwise_dist[idx, :] <= mean_dist
            neigh_mask_near[idx] = False  # Set value at index refering to current example to False.
            
            # Get indices of far neighbours.
            neigh_mask_far = pairwise_dist[idx, :] > mean_dist

            ### /NEIGHBOUR INDICES ###


            ###  HIT AND MISS MASKS ###

            # Get mask of near neighbours with same class.
            hit_neigh_mask_near = np.logical_and(neigh_mask_near, target == target[idx])
            # Get mask of near neighbours with different class.
            miss_neigh_mask_near = np.logical_and(neigh_mask_near, target != target[idx])


            # Get mask of far neighbours with same class.
            hit_neigh_mask_far = np.logical_and(neigh_mask_far, target == target[idx])
            # Get mask of far neighbours with different class.
            miss_neigh_mask_far = np.logical_and(neigh_mask_far, target != target[idx])

            ###  /HIT AND MISS MASKS ###

            
            # Compute probability weights for misses in considered regions.            
            miss_classes = target[miss_neigh_mask_near]
            weights_mult1 = np.empty(miss_classes.size, dtype=np.float)
            u, c = np.unique(miss_classes, return_counts=True)
            neighbour_weights = c/miss_classes.size
            for i, val in enumerate(u):
                weights_mult1[np.where(miss_classes == val)] = neighbour_weights[i]

            miss_classes = target[miss_neigh_mask_far]
            weights_mult2 = np.empty(miss_classes.size, dtype=np.float)
            u, c = np.unique(miss_classes, return_counts=True)
            neighbour_weights = c/miss_classes.size
            for i, val in enumerate(u):
                weights_mult2[np.where(miss_classes == val)] = neighbour_weights[i]

            
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

