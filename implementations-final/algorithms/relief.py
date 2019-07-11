import numpy as np
from scipy.stats import rankdata
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
import numba as nb
import os

from julia import Julia
jl = Julia(compiled_modules=False)

class Relief(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the Relief algorithm

    Kenji Kira, Larry A. Rendell.
    The Feature Selection Problem: Traditional Methods and a New Algorithm.
    
    Author: Jernej Vivod

    """

    # Constructor: initialize learner
    def __init__(self, n_features_to_select=10, m=-1, dist_func=lambda x1, x2: np.sum(np.abs(x1 - x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.m = m                                        # number of examples to sample
        self.dist_func = dist_func  # distance function to use when searching for nearest neighbours
        self.learned_metric_func = learned_metric_func  # learned metric function (is set to None if not using metric learning)

        # Use function written in the Julia programming language to update weights.

        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_relief2.jl")

    def fit(self, data, target):
        """
        Rank features using relief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        # If using a learned metric function
        if self.learned_metric_func != None:
           self.rank, self.weights = self._relief(data, target, self.m, self.dist_func, learned_metric_func=self.learned_metric_func)            
        else:
           self.rank, self.weights = self._relief(data, target, self.m, self.dist_func)

        # Return reference to self
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


    #update_weights: go over features and update weights.
    # @nb.njit
    # def _update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals):
    #     for t in np.arange(data.shape[1]):
    #         # Update weights
    #         weights[t] = weights[t] - (np.abs(e[t] - closest_same[t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))/m + \
    #             (np.abs(e[t] - closest_other[t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))/m

    #     return weights  # Return updated weights


    def _relief(self, data, target, m, dist_func, **kwargs):
        """Compute feature scores using Relief algorithm

        Args: 
            data : Array[np.float64] -- matrix containing examples' data as rows
            target: Array[np.int] -- Matrix containing the examples' class values
            m : int --  Sample size to use when evaluating the feature scores
            dist_func : Callable[[Array[np.float64], Array[np.float64]], Array[np.float64]] -- function for evaluating 
            distances between examples. The function should accept two examples or two matrices of examples and return
            the distance between them.
            **kwargs: can contain argument with key 'learned_metric_func' that maps to a function that accepts a distance
            function and indices of two training examples and returns the distance between the examples in the learned
            metric space.

        Returns:
             Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores
        """

        # Initialize all weights to zero.
        weights = np.zeros(data.shape[1], dtype=np.float)

        # Get maximum and minimum values of each feature
        max_f_vals = np.amax(data, axis=0)
        min_f_vals = np.amin(data, axis=0)

        # Sample m examples without replacement.
        sample_idxs = np.random.choice(np.arange(data.shape[0]), data.shape[0] if m == -1 else m, replace=False)

        # Set m if currently set to signal value -1.
        m = data.shape[0] if m == -1 else m


        # Evaluate features using a sample of m examples.
        for idx in sample_idxs:
            e = data[idx, :]  # Get sampled example data.
            msk = np.array(list(map(lambda x: True if x == target[idx] else False, target)))  # Get mask for examples with same class.

            # Get index of sampled example in subset of examples with same class.
            idx_subset = idx - np.sum(~msk[:idx+1])

            # Find nearest hit and nearest miss.
            if 'learned_metric_func' in kwargs:  # If operating in learned metric space.
                dist = partial(kwargs['learned_metric_func'], dist_func, idx)
                d_same = dist(np.where(msk)[0])
                d_same[idx_subset] = np.inf     # Set distance of sampled example to itself to infinity.
                d_other = dist(np.where(~msk)[0])
                closest_same = data[msk, :][d_same.argmin(), :]
                closest_other = data[~msk, :][d_other.argmin(), :]
            else:                                # Else
                dist = partial(dist_func, e)  # Curry distance function with chosen example data vector.
                d_same = dist(data[msk, :]) 
                d_same[idx_subset] = np.inf  # Set distance of sampled example to itself to infinity.
                d_other = dist(data[~msk, :])
                closest_same = data[msk, :][d_same.argmin(), :]
                closest_other = data[~msk, :][d_other.argmin(), :]

            # ------ weights update ------
            weights = self._update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals)


        # Create array of feature enumerations based on score.
        rank = rankdata(-weights, method='ordinal')
        return rank, weights  # Return vector of feature quality estimates.

