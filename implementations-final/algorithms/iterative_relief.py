import numpy as np
from functools import partial
from scipy.spatial.distance import minkowski
import sklearn.metrics as sk_metrics
from scipy.stats import rankdata
import warnings

from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=UserWarning)

class IterativeRelief(BaseEstimator, TransformerMixin):

    """sklearn compatible implementation of the iterative relief algorithm.

    Bruce Draper, Carol Kaito, Jose Bins.
    Iterative Relief.

    Author: Jernej Vivod
    """

    def __init__(self, n_features_to_select=10,  m=-1, min_incl=3, 
            dist_func=lambda w, x1, x2 : np.sum(np.abs(w*(x1-x2)), 1), max_iter=100, learned_metric_func=None):
        self.m = m  # sample size
        self.min_incl = min_incl  # minimal number of examples from each class to include in hypersphere
        self.dist_func = dist_func  # metric function to measure distance between examples
        self.max_iter = max_iter  # maximal number of iterations
        self.learned_metric_func = learned_metric_func  # learned metric function
        self.n_features_to_select = n_features_to_select  # number of best features to select

    def min_radius(self, n, data, target, dist_metric, mode, **kwargs):
        """
        Compute minimum radius of hypersphere such that for each example in
        the data matrix as the centre the sphere will contain at least n examples from
        same class and n examples from a different class.

        Args:
            n : int -- minimum number of examples from same class and different class a hypersphere with centre in
            each example in the dataset should contain

            data : Array[np.float64] -- Matrix containing examples' features as rows

            target : Array[np.int] Matrix of target variable values

            dist_metric : Callable[[Array[np.float64], Array[np.float64]], np.float64] -- distance metric for distance matrix computation

            mode : str -- equal to 'index' if selecting examples by their index and equal to 'example' if passing in explicit examples.

            **kwargs -- argument with keyword learned_metric_func can contain a learned metric function.

        Returns:
        np.float64 : Minimum acceptable radius of the hypersphere

        """

        # Allocate array for storing minimum acceptable radius for each example in dataset.
        min_r = np.empty(data.shape[0], dtype=np.float)

        # Initialize distance matrix.
        dist_mat = None

        # If operating in learned metric space.
        if mode == "index":
            dist_metric_aux = lambda x1, x2 : dist_metric(np.ones(data.shape[1], dtype=np.float), x1[np.newaxis], x2[np.newaxis])
            dist_func = partial(kwargs['learned_metric_func'], dist_metric_aux)
            dist_func_adapter = lambda x1, x2 : dist_func(np.int(np.where(np.sum(np.equal(x1, data), 1) == data.shape[1])[0][0]), 
                    np.int(np.where(np.sum(np.equal(x2, data), 1) == data.shape[1])[0][0]))
            dist_mat = sk_metrics.pairwise_distances_chunked(data, metric=dist_func_adapter, working_memory=0)
        elif mode == "example":  # else
            dist_func = lambda x1, x2 : dist_metric(np.ones(data.shape[1], dtype=np.float), x1[np.newaxis], x2[np.newaxis])
            dist_mat = sk_metrics.pairwise_distances_chunked(data, metric=dist_func, n_jobs=-1, working_memory=0)
        else:
            raise ValueError('Unknown mode specifier {0}'.format(mode))

        # Go over examples and compute minimum acceptable radius for each example.
        for k in np.arange(data.shape[0]):
            dist_from_e = next(dist_mat)[0]  # Get next row of distances matrix.
            msk = target == target[k]        # Get mask for examples from same class.
            dist_same = dist_from_e[msk]     # Get minimum distance that includes n examples from same class.
            dist_diff = dist_from_e[~msk]    # Get minimum distance that includes n examples from different class.
            try:
                min_r[k] = np.max((np.sort(dist_same)[n], np.sort(dist_diff)[n-1]))  # Compute minimum radius for this example.
            except IndexError:
                raise ValueError('Insufficient examples with class {0} for given value of n (n = {1})'.format(target[k], n))

        return np.max(min_r)  # Return maximum of array of minimum acceptable radiuses for each example

    
    def fit(self, data, target):
        """
        Rank features using relief feature selection algorithm

        Args:
            data : Array[np.float64] -- matrix of examples
            target : Array[np.int] -- vector of target values of examples

        Returns:
            self
        """
        
        # Get number of instances with class that has minimum number of instances.
        _, instances_by_class = np.unique(target, return_counts=True)
        min_instances = np.min(instances_by_class)
       
        # If class with minimal number of examples has less than min_incl examples, issue warning
        # that parameter min_incl was reduced.
        if min_instances < self.min_incl:
            warnings.warn("Parameter k was reduced to {0} because one of the classes " \
                    "does not have {1} instances associated with it.".format(min_instances, self.min_incl), Warning)


        if self.learned_metric_func != None:
            self.rank, self.weights = self._iterative_relief(data, target, self.m, min(min_instances-1, self.min_incl), 
                    self.dist_func, self.max_iter, learned_metric_func=self.learned_metric_func)
        else:
            self.rank, self.weights = self._iterative_relief(data, target, self.m, min(min_instances-1, self.min_incl), 
                    self.dist_func, self.max_iter)

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



    def _iterative_relief(self, data, target, m, min_incl, dist_func, max_iter, **kwargs):
        """Compute feature ranks and scores using Iterative Relief algorithm

        Args:
            data : Array[np.float64] -- Matrix containing examples' features in rows

            target : Array[np.int] Matrix containing the target variable values

            m : int --  Sample size to use when evaluating the feature scores

            min_incl : int -- the minimum number of examples from same and other 
            classes that a hypersphere centered at each examples should contain.

            dist_func : Callable[[Array[np.float64], Array[np.float64], Array[np.float64]], np.float64] -- 
            distance function for evaluating distance between examples. 
            The function should be able to take two matrices of examples and return a vector of distances
            between the examples. The distance function should accept a weights parameter.

            max_iter : int -- Maximum number of iterations to compute


        Returns:
        Array[np.int], Array[np.float64] -- Array of feature enumerations based on the scores, 
                                            array of feature scores

        Author: Jernej Vivod
        """

        # If operating in learned metric space:
        if 'learned_metric_func' in kwargs:
            # Get minimum acceptable radius using learned metric.
            min_r = self.min_radius(min_incl, data, target, dist_func, mode='index', 
                    learned_metric_func=kwargs['learned_metric_func'])
        else:  # else

            # Get minimum acceptable radius.
            min_r = self.min_radius(min_incl, data, target, dist_func, mode='example')

        # Initialize distance weights.  
        dist_weights = np.ones(data.shape[1], dtype=np.float)

        # Initialize iteration counter, convergence indicator and
        # Array for storing feature weights from previous iteration.
        iter_count = 0
        convergence = False
        feature_weights_prev = np.zeros(data.shape[1], dtype=np.float)

        # Iterate
        while iter_count < max_iter and not convergence:
            iter_count += 1   # Increment iteration counter.

            # Reset feature weights to zero and sample examples.
            feature_weights = np.zeros(data.shape[1], dtype=np.float)
            idx_sampled = np.random.choice(data.shape[0], data.shape[0] if m == -1 else m, replace=False)

            # Set m if currently set to signal value -1.
            m = data.shape[0] if m == -1 else m


            # Go over sampled examples.
            for idx in idx_sampled:

                e = data[idx, :]  # Get next sampled example.
                data_filt = np.vstack((data[:idx, :], data[idx+1:, :]))
                target_filt = np.delete(target, idx)

                # Compute inclusion using learned metric function if specified.
                if 'learned_metric_func' in kwargs:
                    dist = partial(kwargs['learned_metric_func'], lambda x1, x2: dist_func(dist_weights, x1, x2), int(idx))

                    # Compute hypersphere inclusions and distances to examples within the hypersphere.
                    # Distances to examples from same class.
                    dist_same_all = dist(np.arange(data_filt.shape[0]))[target_filt == target[idx]]
                    sel = dist_same_all <= min_r
                    dist_same = dist_same_all[sel]
                    data_same = (data_filt[target_filt == target[idx], :])[sel, :]
    
                    # Distances to examples with different class.
                    dist_other_all = dist(np.arange(data_filt.shape[0]))[target_filt != target[idx]]  
                    sel = dist_other_all <= min_r
                    dist_other = dist_other_all[sel]
                    data_other = (data_filt[target_filt != target[idx], :])[sel, :]
                else:
                    # Compute hypersphere inclusions and distances to examples within the hypersphere.
                    # Distances to examples from same class.
                    dist_same_all = dist_func(dist_weights, data_filt[target_filt == target[idx], :], e)
                    sel = dist_same_all <= min_r
                    dist_same = dist_same_all[sel]
                    data_same = (data_filt[target_filt == target[idx]])[sel, :]
                    
                    # Distances to examples with different class.
                    dist_other_all = dist_func(dist_weights, data_filt[target_filt != target[idx], :], e)
                    sel = dist_other_all <= min_r
                    dist_other = dist_other_all[sel]
                    data_other = (data_filt[target_filt != target[idx]])[sel, :]

                # *********** Feature Weights Update ***********
                w_miss = np.maximum(0, 1 - (dist_other**2/min_r**2))
                w_hit = np.maximum(0, 1 - (dist_same**2/min_r**2))

                numerator1 = np.sum(np.abs(e - data_other) * w_miss[np.newaxis].T, 0)
                denominator1 = np.sum(w_miss) + np.finfo(float).eps

                numerator2 = np.sum(np.abs(e - data_same) * w_hit[np.newaxis].T, 0)
                denominator2 = np.sum(w_hit) + np.finfo(float).eps

                feature_weights += numerator1/denominator1 - numerator2/denominator2
                # **********************************************

            # Update distance weights by feature weights - use algorithm's own feature evaluations
            # to weight features when computing distances.
            dist_weights += feature_weights

            # Check convergence.
            if np.sum(np.abs(feature_weights - feature_weights_prev)) < 1.0e-3:
                convergence = True

            feature_weights_prev = feature_weights

        # Rank features and return rank and distance weights.
        rank = rankdata(-dist_weights, method='ordinal')
        return rank, dist_weights

