import numpy as np
import pdb

def relieff(data, m, k, dist_func):
    """Compute feature scores using ReliefF algorithm

    --- Parameters: ---

    data: Matrix containing examples' data as rows (the last column contains the classes)

    m: Sample size to use when evaluating the feature scores

    k: Number of closest examples from each class to use

    dist_func: function for evaluating distances between examples. The function should acept two
    examples or two matrices of examples and return 

    ------

    Returns:
    Array of feature enumerations based on the scores, array of feature scores

    Author: Jernej Vivod

    """

    # Initialize all weights to 0.
    weights = np.zeros(data.shape[1] - 1, dtype=float)

    # Get indices of examples in sample.
    idx_sampled = np.random.choice(np.arange(data.shape[0]), m, replace=False)

    # Get maximum and minimum values of each feature.
    max_f_vals = np.amax(data[:, :-1], 0)
    min_f_vals = np.amin(data[:, :-1], 0)

    # Get all unique classes.
    classes = np.unique(data[:, -1])

    # Get probabilities of classes in training set.
    p_classes = np.vstack(np.unique(data[:, -1], return_counts=True)).T
    p_classes[:, 1] = p_classes[:, 1] / np.sum(p_classes[:, 1])


    # Go over sampled examples' indices.
    for idx in idx_sampled:

        # Get next example.
        e = data[idx, :]

        # Get index of next sampled example in group of examples with same class.
        idx_class = idx - np.sum(data[:idx, -1] != e[-1])
        
        # Find k nearest examples from same class.
        distances_same = dist_func(e[:-1], data[data[:, -1] == e[-1], :-1])
        # Set distance of sampled example to itself to infinity.
        distances_same[idx_class] = np.inf

        # Find closest examples from same class.
        idxs_closest_same = np.argpartition(distances_same, k)[:k] #
        closest_same = (data[data[:, -1] == e[-1], :])[idxs_closest_same, :] #

        # Allocate matrix template for getting nearest examples from other classes.
        closest_other = np.zeros((k * (len(classes) - 1), data.shape[1])) #

        # Initialize pointer for adding examples to template matrix.
        top_ptr = 0
        for cl in classes:  # Go over classes different than the one of current sampled example.
            if cl != e[-1]:

                # Get closest k examples with class cl
                distances_cl = dist_func(e[:-1], data[data[:, -1] == cl, :-1])
                idx_closest_cl = np.argpartition(distances_cl, k)[:k]

                # Add found closest examples to matrix.
                closest_other[top_ptr:top_ptr+k, :] = (data[data[:, -1] == cl, :])[idx_closest_cl, :]
                top_ptr = top_ptr + k


        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 0] != e[-1], 1]
        
        # Compute diff sum weights for closest examples from different class.
        p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == e[-1], 1]) #
        

        # ------ weights update ------

        # Go over features and update weights.
        for t in np.arange(data.shape[1]-1):

            # Penalty term
            penalty = np.sum(abs(e[t] - closest_same[:, t])/(max_f_vals[t] - min_f_vals[t]))

            # Reward term
            reward = np.sum(np.repeat(p_weights, k) * (abs(e[t] - closest_other[:, t])/(max_f_vals[t] - min_f_vals[t])))

            # Weights update
            weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)
            
        # Create array of feature enumerations based on score.
        ranks = np.argsort(weights, 0)[::-1]


    return ranks, weights

# Test
if __name__ == '__main__':

    def minkowski_distance(e1, e2, p):
        return np.sum(np.abs(e1 - e2)**p, 1)**(1/p)

    test_data = np.loadtxt('rba_test_data2.m')

    test_data_large = np.random.rand(200, 17000)
    target = ((np.random.rand(200) > 0.5).astype(int))[np.newaxis].T
    test_data_large = np.hstack((test_data_large, target))

    rank, weights = relieff(test_data, test_data.shape[0], 5, lambda a, b: minkowski_distance(a, b, 2));