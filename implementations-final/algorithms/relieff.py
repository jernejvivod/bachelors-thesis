import numpy as np
import numba as nb
from scipy.stats import rankdata
from functools import partial
import os
import sys
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from julia import Julia
jl = Julia(compiled_modules=False)


class Relieff(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the ReliefF algorithm
    
    Igor Kononenko, Edvard Simec, Marko Robnik-Sikonja.
    Overcoming the myopia of inductive learning algorithms with RELIEFF.
    
    Author: Jernej Vivod
    """
   
    def __init__(self, n_features_to_select=10, m=-1, k=5, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1), learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.m = m                                        # example sample size
        self.k = k                                        # the k parameter
        self.dist_func = dist_func                        # distance function
        self.learned_metric_func = learned_metric_func    # learned distance function

        # Use function written in Julia programming language to update feature weights.
        script_path = os.path.abspath(__file__)
        self._update_weights = jl.include(script_path[:script_path.rfind('/')] + "/julia-utils/update_weights_relieff2.jl")


    def fit(self, data, target):
        """
        Rank features using ReliefF feature selection algorithm

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
        if min_instances < self.k:
            warnings.warn("Parameter k was reduced to {0} because one of the classes " \
                    "does not have {1} instances associated with it.".format(min_instances, self.k), Warning)

        if self.learned_metric_func != None:
            self.rank, self.weights = self._relieff(data, target, self.m, min(self.k, min_instances), 
                    self.dist_func, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._relieff(data, target, self.m, min(self.k, min_instances), 
                    self.dist_func)
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


    # update_weights: go over features and update weights.
    # @nb.njit
    # def _update_weights(self, data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals):

    #     for t in np.arange(data.shape[1]):

    #         # Penalty term
    #         penalty = np.sum(np.abs(e[t] - closest_same[:, t])/((max_f_vals[t] - min_f_vals[t]) + np.finfo(np.float64).eps))

    #         # Reward term
    #         reward = np.sum(weights_mult * (np.abs(e[t] - closest_other[:, t])/((max_f_vals[t] - min_f_vals[t] + np.finfo(np.float64).eps))))

    #         # Weights update
    #         weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)

    #     # Return updated weights.
    #     return weights


    def _relieff(self, data, target, m, k, dist_func, **kwargs):

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

        # Get indices of examples in sample.
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

            # Get index of next sampled example in group of examples with same class.
            idx_class = idx - np.sum(target[:idx] != target[idx])
          
            # If keyword argument with keyword 'learned_metric_func' exists...
            if 'learned_metric_func' in kwargs:

                # Partially apply distance function.
                dist = partial(kwargs['learned_metric_func'], dist_func, np.int(idx))

                # Compute distances to examples from same class in learned metric space.
                distances_same = dist(np.where(target == target[idx])[0])

                # Set distance of sampled example to itself to infinity.
                distances_same[idx_class] = np.inf

                # Find k closest examples from same class.
                idxs_closest_same = np.argpartition(distances_same, k-1)[:k]
                closest_same = (data[target == target[idx], :])[idxs_closest_same, :]
            else:
                # Find k nearest examples from same class.
                distances_same = dist_func(e, data[target == target[idx], :])

                # Set distance of sampled example to itself to infinity.
                distances_same[idx_class] = np.inf

                # Find closest examples from same class.
                idxs_closest_same = np.argpartition(distances_same, k-1)[:k]
                closest_same = (data[target == target[idx], :])[idxs_closest_same, :]

            # Allocate matrix template for getting nearest examples from other classes.
            closest_other = np.empty((k * (len(classes) - 1), data.shape[1]), dtype=np.float)

            # Initialize pointer for adding examples to template matrix.
            top_ptr = 0
            for cl in classes:  # Go over classes different than the one of current sampled example.
                if cl != target[idx]:
                    # If keyword argument with keyword 'learned_metric_func' exists...
                    if 'learned_metric_func' in kwargs:
                        # get closest k examples with class cl if using learned distance metric.
                        distances_cl = dist(np.where(target == cl)[0])
                    else:
                        # Get closest k examples with class cl.
                        distances_cl = dist_func(e, data[target == cl, :])
                    # Get indices of closest exmples from class cl.
                    idx_closest_cl = np.argpartition(distances_cl, k-1)[:k]

                    # Add found closest examples to matrix.
                    closest_other[top_ptr:top_ptr+k, :] = (data[target == cl, :])[idx_closest_cl, :]
                    top_ptr = top_ptr + k


            # Get probabilities of classes not equal to class of sampled example.
            p_classes_other = p_classes[p_classes[:, 0] != target[idx], 1]
           
            # Compute diff sum weights for closest examples from different class.
            p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == target[idx], 1])
            weights_mult = np.repeat(p_weights, k) # Weights multiplier vector
           

            # ------ weights update ------
            weights = np.array(self._update_weights(data, e[np.newaxis], closest_same, closest_other, weights[np.newaxis],
                    weights_mult[np.newaxis].T, m, k, max_f_vals[np.newaxis], min_f_vals[np.newaxis]))


        # Return feature rankings and weights.
        return rankdata(-weights, method='ordinal'), weights

