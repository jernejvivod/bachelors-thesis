import numpy as np
import numba as nb
from functools import partial
import pdb

def relieff(data, target, m, k, dist_func, **kwargs):
    """Compute feature scores using ReliefF algorithm

    --- Parameters: ---

    data: Matrix containing examples' data as rows 

    target: matrix containing the example's target variable value

    m: Sample size to use when evaluating the feature scores

    k: Number of closest examples from each class to use

    dist_func: function for evaluating distances between examples. The function should acept two
	examples or two matrices of examples and return 
    
    **kwargs: can contain argument with key 'learned_metric_func' that maps to a function that accepts a distance
    function and indices of two training examples and returns the distance between the examples in the learned
    metric space.
	------

    Returns:
    Array of feature enumerations based on the scores, array of feature scores

    Author: Jernej Vivod

    """

    # update_weights: go over features and update weights.
    @nb.njit
    def _update_weights(data, e, closest_same, closest_other, weights, weights_mult, max_f_vals, min_f_vals):
        for t in np.arange(data.shape[1]):

            # Penalty term
            penalty = np.sum(np.abs(e[t] - closest_same[:, t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))

            # Reward term
            reward = np.sum(weights_mult * (np.abs(e[t] - closest_other[:, t])/((max_f_vals[t] - min_f_vals[t] + 1e-10))))

            # Weights update
            weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)

        # Return updated weights.
        return weights



    # Initialize all weights to 0.
    weights = np.zeros(data.shape[1], dtype=float)

    # Get indices of examples in sample.
    idx_sampled = np.random.choice(np.arange(data.shape[0]), m, replace=False)

    # Get maximum and minimum values of each feature.
    max_f_vals = np.amax(data[:, :], 0)
    min_f_vals = np.amin(data[:, :], 0)

    # Get all unique classes.
    classes = np.unique(target)
    pdb.set_trace()
    # Get probabilities of classes in training set.
    p_classes = np.vstack(np.unique(target, return_counts=True)).T
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
            dist = partial(kwargs['learned_metric_func'], dist_func, idx)

            # Compute distances to examples from same class in learned metric space.
            distances_same = dist(np.where(target == target[idx])[0])

            # Set distance of sampled example to itself to infinity.
            distances_same[idx_class] = np.inf

            # Find k closest examples from same class.
            idxs_closest_same = np.argpartition(distances_same, k)[:k]
            closest_same = (data[target == target[idx], :])[idxs_closest_same, :]
        else:
            # Find k nearest examples from same class.
            distances_same = dist_func(e, data[target == target[idx], :])

            # Set distance of sampled example to itself to infinity.
            distances_same[idx_class] = np.inf

            # Find closest examples from same class.
            idxs_closest_same = np.argpartition(distances_same, k)[:k] #
            closest_same = (data[target == target[idx], :])[idxs_closest_same, :] #

        # Allocate matrix template for getting nearest examples from other classes.
        closest_other = np.zeros((k * (len(classes) - 1), data.shape[1])) #

        # Initialize pointer for adding examples to template matrix.
        top_ptr = 0
        for cl in classes:  # Go over classes different than the one of current sampled example.
            if cl != target[idx]:
                # If keyword argument with keyword 'learned_metric_func' exists...
                if 'learned_metric_func' in kwargs:
                    # get closest k examples with class cl if using learned distance metric.
                    distances_cl = dist(np.where(target == cl)[0])
                else:
                    # Get closest k examples with class cl
                    distances_cl = dist_func(e, data[target == cl, :])
                # Get indices of closest exmples from class cl
                idx_closest_cl = np.argpartition(distances_cl, k)[:k]

                # Add found closest examples to matrix.
                closest_other[top_ptr:top_ptr+k, :] = (data[target == cl, :])[idx_closest_cl, :]
                top_ptr = top_ptr + k


        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 0] != target[idx], 1]
       
        # Compute diff sum weights for closest examples from different class.
        p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == target[idx], 1])
        weights_mult = np.repeat(p_weights, k) # Weights multiplier vector
        

        # ------ weights update ------
        weights = _update_weights(data, e, closest_same, closest_other, weights, weights_mult, max_f_vals, min_f_vals)
        
        # Create array of feature enumerations based on score.
        rank = np.argsort(weights, 0)[::-1]


    return rank, weights


# Test
if __name__ == '__main__':
    test_data = np.loadtxt('rba_test_data2.m')
    data, target = (lambda x: (x[:, :-1], x[:, -1]))(test_data)
    rank, weights = relieff(data, target, test_data.shape[0], 5, lambda x1, x2: np.sum(np.abs(x1 - x2)**2, 1)**(1/2));
