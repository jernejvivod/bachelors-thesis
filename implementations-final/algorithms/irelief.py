import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from scipy.stats import rankdata

from sklearn.base import BaseEstimator, TransformerMixin

class IRelief(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the i-relief algorithm

    Yijun Sun. Iterative RELIEF for Feature Weighting: Algorithms, Theories, and Applications.

    Author: Jernej Vivod

    """
    
    def __init__(self, n_features_to_select=10, dist_func=lambda w, x1, x2 : np.sum(np.abs(w*(x1-x2))), 
            max_iter=100, k_width=5, conv_condition=1.0e-12, initial_w_div=1, learned_metric_func=None):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.dist_func = dist_func                        # distance function to use
        self.max_iter = max_iter                          # Maximum number of iterations
        self.k_width = k_width                            # kernel width
        self.conv_condition = conv_condition              # convergence condition
        self.initial_w_div = initial_w_div                # initial weight quotient
        self.learned_metric_func = learned_metric_func    # learned metric function

    def fit(self, data, target):
        """
        Rank features using relief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """

        # Run I-RELIEF feature selection algorithm.
        if self.learned_metric_func != None:
            self.rank, self.weights = self._irelief(data, target, self.dist_func, self.max_iter, self.k_width, 
                    self.conv_condition, self.initial_w_div, learned_metric_func=self.learned_metric_func(data, target))
        else:
            self.rank, self.weights = self._irelief(data, target, self.dist_func, self.max_iter, self.k_width, 
                    self.conv_condition, self.initial_w_div)

        # Return reference to self.
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


    def _get_mean_mh_vals(self, data, classes, dist_weights, sig):

        """
        get mean m and mean h values for each example in dataset.

        Args:
            data : Array[np.float64] -- training examples
            classes : Array[np.int] -- class values of examples

        Returns:
            Array[np.float64], Array[np.float64] -- array of average m values for each training example,
            array of average h values for each training example.
        """
       
        # Allocate matrix for storing results.
        mean_h = np.empty(data.shape, dtype=np.float)
        mean_m = np.empty(data.shape, dtype=np.float)

        # Go over rows of pairwise differences.
        for idx in np.arange(data.shape[0]):

            # Compute m values.
            m_vals = np.abs(data[idx, :] - data[classes != classes[idx], :])
            h_vals = np.abs(data[idx, :] - data[np.logical_and(classes == classes[idx], np.arange(data.shape[0]) != idx), :])

            # Compute kernel function values.
            f_m_vals = np.exp(-np.sum(dist_weights*m_vals, axis=1)/sig)
            f_h_vals = np.exp(-np.sum(dist_weights*h_vals, axis=1)/sig)

            # Compute vector of probabilities of misses being nearest misses.
            pm_vec = f_m_vals/np.sum(f_m_vals)
            ph_vec = f_h_vals/np.sum(f_h_vals)

            # Compute mean_m_values for each example
            mean_m[idx, :] = np.sum(pm_vec[np.newaxis].T * m_vals, 0)
            mean_h[idx, :] = np.sum(ph_vec[np.newaxis].T * h_vals, 0)

        return mean_m, mean_h


    def _get_gamma_vals(self, dist_mat, classes, dist_weights, sig):

        """
        For each example get probability of it being an outlier.
        Function depends on weights (through distance matrix).

        Args:
            dist_mat : Array[np.float64] -- distance matrix
            classes : Array[np.int] -- class values of examples
            kern_func : Callable[[np.float64], np.float64]

        Returns:
            Array[np.float64] -- probabilities of examples being outliers
        """

        # Allocate array for storing results.
        po_vals = np.empty(dist_mat.shape[0], dtype=np.float)

        # Go over rows of distance matrix.
        for idx in np.arange(data.shape[0]):

            # Compute probability of n-th example being an outlier.
            m_vals = np.abs(data[idx, :] - data[classes != classes[idx], :])
            d_vals = np.abs(data[idx, :] - data)
            f_m_vals = np.exp(-np.sum(dist_weights*m_vals, axis=1)/sig)
            f_d_vals = np.exp(-np.sum(dist_weights*d_vals, axis=1)/sig)
            po_vals[idx] = np.sum(f_m_vals)/np.sum(f_d_vals)
        
        # Gamma values are probabilities of examples being inliers.
        return 1 - po_vals
        


    def _get_nu(self, gamma_vals, mean_m_vals, mean_h_vals, nrow):
        """
        get nu vector (See article, pg. 4)

        Args:
            gamma_vals : Array[np.float64] -- gamma values of each example
            mean_m_vals : Array[np.float64] -- mean m value of each example
            mean_h_vals : Array[np.float64] -- mean h value of each example
            nrow : np.int -- number of rows in each example

        Returns:
            Array[np.float64] -- the nu vector
        """
        return (1/nrow) * np.sum(gamma_vals[np.newaxis].T * (mean_m_vals - mean_h_vals), 0)


    def _irelief(self, data, target, dist_func, max_iter, k_width, conv_condition, initial_w_div, **kwargs):
        """
        Implementation of the I-Relief algoritm as described by Yiun et al.

        Args:
            data : Array[np.float64] -- training examples
            target: Array[np.int] -- examples' class values
            dist_func : Callable[[np.floatt64, Array[np.float64]], np.float64] -- weighted distance function 
            max_iter: maximum number of iterations to perform
            k_width : np.float64 -- kernel width (used in gamma values computation)
            conv_condition : np.float64 -- threshold for convergence declaration (if change in weights < conv_condition, stop running the iteration loop)
            initial_w_div : np.float64 -- value with which to divide the initial weights values
            kwargs -- can contain keyword argument 'learned_metric_func' which contains the learned metric function (takes two indices)

        Returns:
            Array[np.int], Array[np.float64] -- array of feature rankings and array of feature scores
        """

        # Intialize convergence indicator and distance weights for features.
        convergence = False 
        dist_weights = np.ones(data.shape[1], dtype=np.float)/initial_w_div


        # Initialize iteration counter.
        iter_count = 0

        ### Main iteration loop. ###
        while iter_count < max_iter and not convergence: 

            # weighted distance function
            dist_func_w = partial(dist_func, dist_weights) 

            # Compute weighted pairwise distances.
            if 'learned_metric_func' in kwargs:
                dist_func_w_learned = partial(kwargs['learned_metric_func'], dist_func_w)
                pairwise_dist = self._get_pairwise_distances(data, dist_func_w_learned, mode="index")
            else:
                # Get weighted distance function.
                pairwise_dist = self._get_pairwise_distances(data, dist_func_w, mode="example")

            # Get gamma values and compute nu.
            gamma_vals = self._get_gamma_vals(pairwise_dist, target, dist_weights, sig=3)
            
            # Get mean m and mean h vals for all examples.
            mean_m_vals, mean_h_vals = self._get_mean_mh_vals(data, target, dist_weights, sig=3)
            
            # Get nu vector.
            nu = self._get_nu(gamma_vals, mean_m_vals, mean_h_vals, data.shape[0]) 

            # Update distance weights.
            dist_weights_nxt = np.clip(nu, a_min=0, a_max=None)/np.linalg.norm(np.clip(nu, a_min=0, a_max=None))

            # Check if convergence criterion satisfied. If not, continue with next iteration.
            if np.sum(np.abs(dist_weights_nxt - dist_weights)) < conv_condition:
                dist_weights = dist_weights_nxt
                convergence = True
            else:
                dist_weights = dist_weights_nxt
                iter_count += 1

        ############################

        # Rank features by feature weights.
        rank = rankdata(-dist_weights, method='ordinal')

        # Return feature ranks and last distance weights.
        return rank, dist_weights


if __name__ == '__main__':
    import scipy.io as sio
    data = sio.loadmat('data.mat')['data']
    target = np.ravel(sio.loadmat('target.mat')['target'])
    irelief = IRelief() 
    irelief.fit(data, target)
    print(irelief.weights)
