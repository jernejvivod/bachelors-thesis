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

    # Constructor: initialize learner.
    def __init__(self, n_features_to_select=10, m=-1, dist_func=lambda x1, x2: np.sum(np.abs(x1 - x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.m = m
        self.dist_func = dist_func
        script_path = os.path.abspath(__file__)
        self._relief = jl.include(script_path[:script_path.rfind('/')] + "/relief2.jl")


    def fit(self, data, target):

        """
        Rank features using relief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """
        
        # Compute feature weights and rank.
        self.weights = self._relief(data, target, self.m, self.dist_func)
        self.rank = rankdata(-self.weights, method='ordinal')

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


if __name__ == '__main__':
    import scipy.io as sio
    data = sio.loadmat('../datasets/final6/ovariancancer/data.mat')['data']
    target = np.ravel(sio.loadmat('../datasets/final6/ovariancancer/target.mat')['target'])
    script_path = os.path.abspath(__file__)
    get_dist_func = jl.include(script_path[:script_path.rfind('/')] + "/augmentations/me_dissim2.jl")
    dist_func = get_dist_func(10, data)

    import pdb
    pdb.set_trace()

    relief = Relief(dist_func=dist_func)
    relief.fit(data, target)
    print(relief.weights)



