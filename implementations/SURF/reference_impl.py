from __future__ import print_function
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from .relieff import ReliefF
from .scoring_utils import SURF_compute_scores


class SURF(ReliefF):

    """Feature selection using data-mined expert knowledge.

    Based on the SURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

    def __init__(self, n_features_to_select=10, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to
            retain after feature selection is applied.
        discrete_threshold: int (default: 10)
            Value used to determine if a feature is discrete or continuous.
            If the number of unique levels in a feature is > discrete_threshold, then it is
            considered continuous, or discrete otherwise.
        verbose: bool (default: False)
            If True, output timing of distance array and scoring
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.

        """
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

############################# SURF ############################################
    def _find_neighbors(self, inst, avg_dist):
        """ Identify nearest hits and misses within radius defined by average distance over whole distance array.
        This works the same regardless of endpoint type. """
        NN = []  # Get nearest neighbors.
        min_indicies = []

        for i in range(self._datalen):  # Go over examples.
            if inst != i:
                locator = [inst, i]
                if i > inst:
                    locator.reverse()
                d = self._distance_array[locator[0]][locator[1]]
                if d < avg_dist:  # if neighbour less than average distance away.
                    min_indicies.append(i)  # Save its index.
        for i in range(len(min_indicies)):  # Go over marked neighbors.
            NN.append(min_indicies[i])  # Append indices to NN list.
        return np.array(NN, dtype=np.int32)  # Return numpy array of nearest neighbour indices.

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield SURF scores. """
        sm = cnt = 0  # Define commulative counters of distance and length.
        for i in range(self._datalen):
            sm += sum(self._distance_array[i])
            cnt += len(self._distance_array[i])
        avg_dist = sm / float(cnt)  # Compute average distance.

        nan_entries = np.isnan(self._X)  # Get nan entries.

        NNlist = [self._find_neighbors(datalen, avg_dist) for datalen in range(self._datalen)]  # Get nearest neighbors.

        # Compute SURF scores.
        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            SURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                 NN, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN in zip(range(self._datalen), NNlist)), axis=0)

        return np.array(scores)
