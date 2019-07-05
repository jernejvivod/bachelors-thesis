import numpy as np
import numba as nb
from scipy.stats import rankdata
from functools import partial

import os
import sys

from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)


class SWRFStar(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the ReliefF algorithm

    Matthew E Stokes, Shyam Visweswaran. 
    Application of a spatially-weighted Relief algorithm for ranking genetic predictors of disease.
    
        Author: Jernej Vivod
    """
   
    def __init__(self, n_features_to_select=10, m=-1, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.m = m                                        # number of examples to sample
        self.dist_func = dist_func                        # distance function
        self.learned_metric_func = learned_metric_func    # learned metric function

        # Use function written in Julia programming language to update feature weights.
        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_swrfstar2.jl")


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
            self.rank, self.weights = self._swrfstar(data, target, self.m, self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._swrfstar(data, target, self.m, self.dist_func)
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
        self.fit(data, target)  # Fit data
        return self.transform(data)  # Perform feature selection


    def _swrfstar(self, data, target, m, dist_func, **kwargs):

        """Compute feature scores using ReliefF algorithm

        Args:
            data : Array[np.float64] -- matrix containing examples' data as rows 
            target : Array[np.int] -- matrix containing the example's target variable value
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

        # Get indices of examples in sample of examples.
        idx_sampled = np.random.choice(np.arange(data.shape[0]), data.shape[0] if m == -1 else m, replace=False)
        
        # Set m if currently set to signal value -1.
        m = data.shape[0] if m == -1 else m

        # Get maximum and minimum values of each feature.
        max_f_vals = np.amax(data, 0)
        min_f_vals = np.amin(data, 0)

        # Get all unique classes.
        classes = np.unique(target)

        # Get probabilities of classes in training set.
        p_classes = (np.vstack(np.unique(target, return_counts=True)).T).astype(np.float)
        p_classes[:, 1] = p_classes[:, 1] / np.sum(p_classes[:, 1])

        # Go over sampled examples' indices.
        for idx in idx_sampled:

            # Get next example.
            e = data[idx, :]

            # mask for selecting examples with same class
            # that exludes currently sampled example.
            same_sel = target == target[idx]
            same_sel[idx] = False
            
            # examples with same class value
            same = data[same_sel, :]

            # examples with different class value
            other = data[target != target[idx], :]
            
            # class values of examples with different class value
            target_other = target[target != target[idx]]

            # If keyword argument with keyword 'learned_metric_func' exists...
            if 'learned_metric_func' in kwargs:

                # Partially apply distance function.
                dist = partial(kwargs['learned_metric_func'], dist_func, np.int(idx))

                # Compute distances to examples from same class in learned metric space.
                distances_same = dist(np.where(same_sel)[0])
            else:
                # Compute distances to examples with same class value.
                distances_same = dist_func(e, same)

            # Compute t and u parameter values.
            t_same = np.mean(distances_same)
            u_same = np.std(distances_same)

            if 'learned_metric_func' in kwargs:
                # Compute distances to examples from different class in learned metric space.
                distances_other = dist(np.where(target != target[idx])[0])
            else:
                # Compute distances to examples with different class value.
                distances_other = dist_func(e, other)

            # Compute t and u parameter values.
            t_other = np.mean(distances_other)
            u_other = np.std(distances_other)


            # Compute weights for examples from same class.
            neigh_weights_same = 2.0/(1 + np.exp(-(t_same-distances_same)/(u_same/4.0 + 1e-10)))
            
            # Compute weights for examples from different class.
            neigh_weights_other = 2.0/(1 + np.exp(-(t_other-distances_other)/(u_other/4.0 + 1e-10)))


            # Get probabilities of classes not equal to class of sampled example.
            p_classes_other = p_classes[p_classes[:, 0] != target[idx], 1]
            
            # Get other classes.
            classes_other = p_classes[p_classes[:, 0] != target[idx], 0]
           
            # Compute diff sum weights for examples from different classes.
            p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == target[idx], 1])
            
            # Map weights to 'other' vector and construct weights vector.
            weights_map = np.vstack((classes_other, p_weights))
            
            # Construct weights multiplication vector.
            weights_mult = np.array([weights_map[1, np.where(weights_map[0, :] == t)[0][0]] for t in target_other])

            # ------ weights update ------
            weights = self._update_weights(data, e[np.newaxis], same, other, weights[np.newaxis], 
                    weights_mult[np.newaxis].T, neigh_weights_same[np.newaxis].T, neigh_weights_other[np.newaxis].T, 
                    m, max_f_vals[np.newaxis], min_f_vals[np.newaxis])
       

        # Create array of feature enumerations based on score.
        rank = rankdata(-weights, method='ordinal')
        return rank, weights

