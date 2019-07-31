import numpy as np
import numba as nb
from scipy.stats import rankdata
from functools import partial
from nptyping import Array
from sklearn.metrics import pairwise_distances

from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)

import os


class MultiSURF(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the MultiSURF algorithm

    Ryan J. Urbanowicz, Randal S. Olson, Peter Schmitt, Melissa Meeker, Jason H. Moore.
    Benchmarking Relief-Based Feature Selection Methods for Bioinformatics Data Mining.

    author: Jernej Vivod
    """

    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2)), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # Number of features to select.
        self.dist_func = dist_func                        # distance function to use
        self.learned_metric_func = learned_metric_func    # learned metric function

        # Use function written in Julia programming language to update feature weights.
        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_multisurf2.jl")



    def fit(self, data, target):
        """
        Rank features using MultiSURF feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """
       
        # Run MultiSURF feature selection algorithm.
        if self.learned_metric_func != None:
            self.rank, self.weights = self._multiSURF(data, target, self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._multiSURF(data, target, self.dist_func)
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
        self.fit(data, target)  # Fit data.
        return self.transform(data)  # Perform feature selection.


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

        if mode == "index":
            # Allocate matrix for distance matrix and compute distances.
            dist_func_adapter = lambda x1, x2 : dist_func(np.int(np.where(np.sum(np.equal(x1, data), 1) == data.shape[1])[0][0]),
                    np.int(np.where(np.sum(np.equal(x2, data), 1) == data.shape[1])[0][0]))
            return pairwise_distances(data, metric=dist_func_adapter)
        elif mode == "example":
           return pairwise_distances(data, metric=dist_func)
        else:
            raise ValueError("Unknown mode specifier")


    def _critical_neighbours(self, ex_idx : int, dist_mat : Array[np.float64]) -> Array[np.float64]:
        """
        Find neighbours of instance with index inst_idx in radius defined by average distance to other examples and the standard deviation
        of the distances to other examples.

        Args:
            inst_idx : int -- index of the example
            dist_mat : Array[np.float64] -- pairwise distance matrix for examples
        Returns:
            Array[np.int] -- indices of examples that are considered near neighbors of example with index ex_idx.
        """

        # Get average distance to example with index ex_idx.
        ex_avg_dist : np.float64 = np.mean(dist_mat[ex_idx, np.arange(dist_mat.shape[1]) != ex_idx])
        # Get half of standard deviation of distances to example with index ex_idx.
        ex_d : np.float64 = np.std(dist_mat[ex_idx, np.arange(dist_mat.shape[1]) != ex_idx]) / 2.0
        # Get threshold for near neighbours - half a standard deviation away from mean.
        near_thresh : np.float64 = ex_avg_dist - ex_d

        # Return indices of examples that are considered near neighbours. 
        msk_near = dist_mat[ex_idx, :] < near_thresh
        msk_near[ex_idx] = False
        return np.nonzero(msk_near)[0]


    def _multiSURF(self, data, target, dist_func, **kwargs):

        """Compute feature scores using multiSURF algorithm

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


        # Compute weighted pairwise distances.
        if 'learned_metric_func' in kwargs:
            dist_func_learned = partial(kwargs['learned_metric_func'], dist_func)
            pairwise_dist = self._get_pairwise_distances(data, dist_func_learned, mode="index")
        else:
            pairwise_dist = self._get_pairwise_distances(data, dist_func, mode="example")

        # Get maximum and minimum values of each feature.
        max_f_vals = np.amax(data, 0)
        min_f_vals = np.amin(data, 0)

        # Get all unique classes.
        classes = np.unique(target)

        # Get probabilities of classes in training set.
        p_classes = (np.vstack(np.unique(target, return_counts=True)).T).astype(np.float)
        p_classes[:, 1] = p_classes[:, 1]/np.sum(p_classes[:, 1])

        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.float)

        # Compute hits ans misses for each examples (within radius).
        # first row represents the indices of neighbors within threshold and the second row
        # indicates whether an examples is a hit or a miss.
        neighbours_map = dict.fromkeys(np.arange(data.shape[0]))
        for ex_idx in np.arange(data.shape[0]):
            r1 = self._critical_neighbours(ex_idx, pairwise_dist)  # Compute indices of neighbours.
            r2 = target[r1] == target[ex_idx]  # Compute whether neighbour represents a hit or miss.
            neighbours_map[ex_idx] = np.vstack((r1, r2)) # Add info about neighbours to dictionary. First row represents neighbour indices.
                                                         # Second row represents whether neighbour is a hit or miss (logical 0 or 1).

        # Go over all hits and misses and update weights.
        for ex_idx, neigh_data in neighbours_map.items():

            # Get probabilities of classes not equal to class of sampled example.
            p_classes_other = p_classes[p_classes[:, 0] != target[ex_idx], :]
            p_weights = p_classes_other[:, 1]/(1 - p_classes[p_classes[:, 0] == target[ex_idx], 1])

            # Get classes of miss neighbours.
            classes_other = (target[neigh_data[0]])[np.logical_not(neigh_data[1])]
            
            # Compute probability weights for misses in considered regions.            
            weights_mult = np.empty(classes_other.size, dtype=np.float)
            u, c = np.unique(classes_other, return_counts=True)
            neighbour_weights = c/classes_other.size
            for i, val in enumerate(u):
                weights_mult[np.where(classes_other == val)] = neighbour_weights[i]

            # Update weights.
            weights = self._update_weights(data, data[ex_idx, :][np.newaxis], (data[neigh_data[0, :], :])[neigh_data[1, :], :],
                    (data[neigh_data[0, :], :])[np.logical_not(neigh_data[1, :]), :], weights[np.newaxis], 
                    weights_mult[np.newaxis].T, max_f_vals[np.newaxis], min_f_vals[np.newaxis])

        # Rank weights and return.
        # Create array of feature enumerations based on score.
        rank = rankdata(-weights, method='ordinal')
        return rank, weights

