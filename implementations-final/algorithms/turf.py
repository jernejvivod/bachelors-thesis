class TURF(baseEstimator, transformerMixin):

    """sklearn compatible implementation of the TURF algorithm

        Author: Jernej Vivod
    """

    def __init__(self, n_features_to_select, num_it, rba):
        self._n_features_to_select = n_features_to_select
        self._num_it = num_it
        self._rba = rba


    def fit(self, data, target):

        """
        Rank features using TURF feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        self.rank, self.weights = self._turf(data, target, self._num_it, self._rba)


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
        
        # number of features with lowest weights to remove in each iteration (num_iterations/a).
        feat_to_remove = np.int(np.ceil(np.float(num_it)/np.float(data.shape[1])))

        # Indices of features weighted by the final weights in the original data matrix.
        sel_final = np.arange(data.shape[1])
       
        # Initialize feature weights.
        weights = np.zeros(data.shape[1])
        
        # iteration loop
        for it in np.arange(num_it):

            # Fit rba.
            rba = rba.fit(data, target)

            # Rank features.
            rank_nxt = rba.rank

            ### Remove num_it/a features with lowest weights. ###
            sel = rank_nxt <= rank_nxt.shape[0] - num_to_remove
            ind_sel = np.where(sel)                 # Get indices of kept features.
            ind_rm = np.where(np.logical_not(sel))  # Get indices of removed features.
            ind_rm_original = sel_final[ind_rm]     # Get indices of removed features in original data matrix.
            weights[ind_rm_original] = rba.weights[ind_rm]  # Add weights of discarded features to weights vector.
            sel_final = sel_final[ind_sel]          # Filter set of final selection indices.
            data_filtered = data[:, sel]            # Filter data matrix.
            #####################################################

        # Get final ranking and weights.
        weights_final = rba.weights
        weights[sel_final] = weights_final
        return rankdata(-weights, method='ordinal'), weights

