from __future__ import print_function
import numpy as np
from .surfstar import SURFstar
from .scoring_utils import MultiSURFstar_compute_scores
from sklearn.externals.joblib import Parallel, delayed


class MultiSURFstar(SURFstar):

    """Feature selection using data-mined expert knowledge.

    Based on the MultiSURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

############################# MultiSURF* ########################################
    def _find_neighbors(self, inst):
        """ Identify nearest as well as farthest hits and misses within radius defined by average distance and standard deviation of distances from target instanace.
        This works the same regardless of endpoint type. """
        dist_vect = []  # Define array for storing critical neighbours.
        for j in range(self._datalen):  # Go over examples.
            if inst != j:   # If example not equal to itself
                locator = [inst, j]  # Define locator pair and sort it.
                if inst < j:
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])

        dist_vect = np.array(dist_vect)  # Convert to numpy array.
        inst_avg_dist = np.average(dist_vect)  # Get average distance of neighbour.
        inst_std = np.std(dist_vect) / 2.  # Compute standard deviation of distances.
        near_threshold = inst_avg_dist - inst_std  # Get threshold to consider examples near.
        far_threshold = inst_avg_dist + inst_std  # Get threshold to consider examples far.

        # Define arrays for storing nearest neighbors.
        NN_near = []
        NN_far = []
        for j in range(self._datalen):  # Go over examples.
            if inst != j:  # If example not equal to itself
                locator = [inst, j]  # Define locator pair and sort it.
                if inst < j:
                    locator.reverse()
                if self._distance_array[locator[0]][locator[1]] < near_threshold:
                    NN_near.append(j)
                elif self._distance_array[locator[0]][locator[1]] > far_threshold:
                    NN_far.append(j)

        return np.array(NN_near), np.array(NN_far)

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield MultiSURF* scores. """
        nan_entries = np.isnan(self._X)

        NNlist = [self._find_neighbors(datalen) for datalen in range(self._datalen)]  # Get list of nearest neighbors.
        NN_near_list = [i[0] for i in NNlist] # Get near neighbors of all examples.
        NN_far_list = [i[1] for i in NNlist]  # Get far neighbors of all examples.

        # Compute MultiSURF* scores.
        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            MultiSURFstar_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                          NN_near, NN_far, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN_near, NN_far in zip(range(self._datalen), NN_near_list, NN_far_list)), axis=0)

        return np.array(scores)
