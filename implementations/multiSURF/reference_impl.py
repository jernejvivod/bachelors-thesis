from __future__ import print_function
import numpy as np
from .surfstar import SURFstar
from sklearn.externals.joblib import Parallel, delayed
from .scoring_utils import MultiSURF_compute_scores


class MultiSURF(SURFstar):

    """Feature selection using data-mined expert knowledge.

    Based on the MultiSURF algorithm as introduced in:

    Moore, Jason et al. Multiple Threshold Spatially Uniform ReliefF
    for the Genetic Analysis of Complex Human Diseases.

    """

############################# MultiSURF ########################################
    def _find_neighbors(self, inst):
        """ Identify nearest hits and misses within radius defined by average distance and standard deviation around each target training instance.
        This works the same regardless of endpoint type. """
        dist_vect = []   # Allocate array of distances.
        for j in range(self._datalen):  # Go over examples.
            if inst != j  # If instance not equal to itself.
                locator = [inst, j]  # Define locator pair.
                if inst < j:   # sort pair in a descending fashion.
                    locator.reverse()
                dist_vect.append(self._distance_array[locator[0]][locator[1]])  # append distance

        dist_vect = np.array(dist_vect)  # Conver to numpy array.
        inst_avg_dist = np.average(dist_vect)  # Get average distance.
        inst_std = np.std(dist_vect) / 2.  # Get standard deviation.
        # Defining a narrower radius based on the average instance distance minus the standard deviation of instance distances.
        near_threshold = inst_avg_dist - inst_std  # Define threshold for an example to be considered near.

        # Allocate array for storing near instances.
        NN_near = []
        for j in range(self._datalen):  # Go over examples.
            if inst != j:   # If instance not equal to itself.
                locator = [inst, j] # Define and sort locators.
                if inst < j:
                    locator.reverse()
                if self._distance_array[locator[0]][locator[1]] < near_threshold:  # If distance between examples less than threshold, append.
                    NN_near.append(j)

        return np.array(NN_near)

    def _find_neighbors2(self, inst_idx, dist_mat):
        msk = np.arange(dist_mat.shape[1]) != inst_idx  # Get mask that excludes inst_idx.
        inst_avg_dist = np.average(dist_mat[inst_idx, msk])  # Get average distance to example with index inst_idx.
        inst_std = np.std(dist_mat[inst_idx, msk]) / 2.0  # Get standard deviation of distances to example with index inst_idx.
        near_thresh = inst_avg_dist - inst_std  # Get threshold for near neighbors.
        return np.nonzero(dist_mat[inst_idx, msk] < near_thresh)  # Return indices of examples that are considered near neighbors. 


    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield MultiSURF scores. """
        nan_entries = np.isnan(self._X)

        NNlist = [self._find_neighbors(datalen) for datalen in range(self._datalen)]  # Find neighbors of instances.

        # Compute MultiSURF scores.
        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            MultiSURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                      NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN_near in zip(range(self._datalen), NNlist)), axis=0)

        return np.array(scores)  # Return MultiSURF scores.
