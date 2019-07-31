import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import rankdata
from algorithms.relieff import Relieff

class TuRF(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the TURF algorithm

    Jason H. Moore, Bill C. White.
    Tuning ReliefF for Genome-Wide Genetic Analysis.

    Author: Jernej Vivod
    """

    def __init__(self, n_features_to_select=10, num_it=50, rba=Relieff()):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.num_it = num_it                              # number of iterations to perform
        self.rba = rba                                    # relief-based algorithm to use


    def fit(self, data, target):

        """
        Rank features using TURF feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """
        
        # Run TuRF algorithm.
        self.rank, self.weights = self._turf(data, target, self.num_it, self.rba)
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


    def _turf(self, data, target, num_it, rba):

        """Compute feature scores using TURF algorithm

        Args:
            data : Array[np.float64] -- matrix containing examples' data as rows
            target : Array[np.int] -- matrix containing the example's target variable value
            num_it : int -- number of iterations of feature elimination to perform
            rba -- initialized relief based feature selection algorithm implementation

        Returns:
            Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, array of feature scores

        """
        

        # Indices of features weighted by the final weights in the original data matrix.
        sel_final = np.arange(data.shape[1])
       
        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.float)
        rank = np.empty(data.shape[1], dtype=np.float)

        # set data_filtered equal to initial data.
        data_filtered = data
    
        # Flag to stop iterating if number of features to be removed becomes
        # larger than number of features left in dataset.
        stop_iterating = False

        # Compute value to add to local ranking to get global ranking.
        rank_add_val = data.shape[1]
        
        # iteration loop
        it_idx = 0
        while it_idx < num_it and not stop_iterating:
            it_idx += 1


            # Fit rba.
            rba = rba.fit(data_filtered, target)

            # Rank features.
            rank_nxt = rba.rank

            # number of features with lowest weights to remove in each iteration (num_iterations/a).
            num_to_remove = np.int(np.ceil(np.float(num_it)/np.float(data_filtered.shape[1])))
            
            # If trying to remove more features than present in dataset, remove remaining features and stop iterating.
            if num_to_remove > data_filtered.shape[1]:
                num_to_remove = data_filtered.shape[1]
                stop_iterating = True
            

            ### Remove num_it/a features with lowest weights. ###

            sel = rank_nxt <= rank_nxt.shape[0] - num_to_remove
            ind_sel = np.where(sel)[0]                 # Get indices of kept features.
            ind_rm = np.where(np.logical_not(sel))[0]  # Get indices of removed features.
            ind_rm_original = sel_final[ind_rm]        # Get indices of removed features in original data matrix.
            weights[ind_rm_original] = rba.weights[ind_rm]   # Add weights of discarded features to weights vector.
            rank_rm = rankdata(-rba.weights[ind_rm], method='ordinal')  # Get local ranking of removed features.
            rank_add_val -= num_to_remove                               # Adjust value that converts local ranking to global ranking.
            rank[ind_rm_original] = rank_rm + rank_add_val              # Add value to get global ranking of removed features.
            sel_final = sel_final[ind_sel]                   # Filter set of final selection indices.
            data_filtered = data_filtered[:, sel]            # Filter data matrix.

            #####################################################


        # Get and return final rankings and weights.
        weights_final = np.delete(rba.weights, ind_rm)
        rank_final = rankdata(-weights_final, method='ordinal')
        rank[sel_final] = rank_final
        weights[sel_final] = weights_final
        return rank, weights


if __name__ == '__main__':
    data = np.random.rand(10, 3)
    target = (data[:, 0] > data[:, 1]).astype(np.int)
    turf = TuRF()
    turf.fit(data, target)


