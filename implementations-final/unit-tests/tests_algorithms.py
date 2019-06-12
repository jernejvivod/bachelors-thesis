import unittest
from functools import partial
import numpy as np

## RELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ##########

from algorithms.relief import Relief

class TestRelief(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        relief = Relief()
        self.assertEqual(relief.n_features_to_select, 10)
        self.assertEqual(relief.m, -1)
        self.assertNotEqual(relief.dist_func, None)
        self.assertEqual(relief.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        relief = Relief(n_features_to_select=15, m=80, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(relief.n_features_to_select, 15)
        self.assertEqual(relief.m, 80)
        self.assertNotEqual(relief.dist_func, None)
        self.assertNotEqual(relief.learned_metric_func, None)

    # Test update of feature weights.
    def test_weights_update(self):
        relief = Relief()

        # Initialize parameter values.
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [7.03015, 9.24269, 3.02136],
                         [8.95009, 8.52854, 0.16166],
                         [3.41438, 4.03548, 7.88157],
                         [2.01185, 0.84564, 6.16909],
                         [2.79316, 1.71541, 2.97578],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 2, 1, 1, 3, 3, 3, 1])
        e = data[2, :]  # The third example
        closest_same = data[3, :]  # Closest example from same class
        closest_other = data[9, :]  # Closest example from different class
        weights = np.ones(data.shape[1])  # Current feature weights
        m = data.shape[0]  # Number of examples to sample
        max_f_vals = np.max(data, 0)  # Max value of each feature
        min_f_vals = np.min(data, 0)  # Min value of each feature

        # Compute weights update
        res = np.round(relief._update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals), 5)

        # Result computed by hand.
        correct_res = np.round(np.array([1.004167277552613, 1.0057086828870614, 1.01971232778099]), 5)
        self.assertSequenceEqual(res.tolist(), correct_res.tolist())

    # Test relief algorithm
    def test_relief(self):

        # Training examples
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [7.03015, 9.24269, 3.02136],
                         [4.77481, 8.01036, 7.57880]])

        # Class values
        target = np.array([1, 2, 2, 1])

        relief = Relief(n_features_to_select=2, m=data.shape[0])
        relief = relief.fit(data, target)

        # Get results of method.
        res_rank = relief.rank
        res_weights = np.round(relief.weights, 5)

        # Results computed by hand.
        correct_res_rank = np.array([1, 2, 3])
        correct_res_weights = np.round(np.array([0.11295758, -0.23107757, -0.32095185]), 5)

        self.assertSequenceEqual(res_rank.tolist(), correct_res_rank.tolist())
        self.assertSequenceEqual(res_weights.tolist(), correct_res_weights.tolist())
        

########################################################



## RELIEFF ALGORITHM IMPLEMENTATION UNIT TESTS #########


from algorithms.relieff import Relieff

class TestRelieff(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        relieff = Relieff()
        self.assertEqual(relieff.n_features_to_select, 10)
        self.assertEqual(relieff.m, -1)
        self.assertEqual(relieff.k, 5)
        self.assertNotEqual(relieff.dist_func, None)
        self.assertEqual(relieff.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        relieff = Relieff(n_features_to_select=15, m=80, k=3, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(relieff.n_features_to_select, 15)
        self.assertEqual(relieff.m, 80)
        self.assertEqual(relieff.k, 3)
        self.assertNotEqual(relieff.dist_func, None)
        self.assertNotEqual(relieff.learned_metric_func, None)

    # Test update of feature weights.
    def test_weights_update(self):
        relieff = Relieff(k=2)

        # Initialize parameter values.
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [7.03015, 9.24269, 3.02136],
                         [8.95009, 8.52854, 0.16166],
                         [3.41438, 4.03548, 7.88157],
                         [2.01185, 0.84564, 6.16909],
                         [2.79316, 1.71541, 2.97578],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 2, 1, 1, 3, 3, 3, 1])
        e = data[2, :]  # The third example.
        closest_same = np.vstack((data[3, :], data[1, :]))  # Closest examples from same class
        closest_other = np.vstack((data[9, :], data[4, :], data[6, :], data[8, :]))  # Closest example from different classes
        weights = np.zeros(data.shape[1])  # Feature weights
        weights_mult = np.array([0.4/0.7, 0.4/0.7, 0.3/0.7, 0.3/0.7])  # Weights multipliers
        m = data.shape[0]  # number of examples to sample
        k = 2  # Number of examples from each class to take.
        max_f_vals = np.max(data, 0)  # Max value of each feature
        min_f_vals = np.min(data, 0)  # Min value of each feature
        
        # Compute weights update
        res = np.round(relieff._update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals), 5)

        # Result computed by hand.
        correct_res = np.round(np.array([ 0.01648752,  0.03281824, -0.01643311]), 5)
        self.assertSequenceEqual(res.tolist(), correct_res.tolist())

    # Test relieff algorithm
    def test_relieff(self):
        pass

#########################################################
#
#
#
### ITERATIVE RELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ###
#
#
#from algorithms.iterative_relief import IterativeRelief
#
#class TestIterativeRelief(unittest.TestCase):
#
#    # Test initialization with default parameters.
#    def test_init_default(self):
#        iterative_relief = IterativeRelief()
#        self.assertEqual(iterative_relief.n_features_to_select, 10)
#        self.assertEqual(iterative_relief.m, 100)
#        self.assertEqual(iterative_relief.min_incl, 3)
#        self.assertNotEqual(relieff.dist_func, None)
#        self.assertEqual(iterative_relief.max_iter, 5)
#        self.assertEqual(relieff.learned_metric_func, None)
#
#    # Test initialization with explicit parameters.
#    def test_init_custom(self):
#        iterative_relief = IterativeRelief(n_features_to_select=15, m=80, min_incl=5, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
#        self.assertEqual(iterative_relief.n_features_to_select, 15)
#        self.assertEqual(relieff.m, 80)
#        self.assertEqual(relieff.min_incl, 5)
#        self.assertNotEqual(relieff.dist_func, None)
#        self.assertNotEqual(relieff.learned_metric_func, None)
#    
#    # Test minimal radius for specified number of hypersphere inclusions using distance computations
#    # with explicit examples.
#    def test_min_radius_example(self):
#        iterative_relief = IterativeRelief()
#        
#        # Set parameters
#        n = 3
#        data = [[2.09525, 0.26961, 3.99627]
#                [9.86248, 6.22487, 8.77424]
#                [7.03015, 9.24269, 3.02136]
#                [8.95009, 8.52854, 0.16166]
#                [3.41438, 4.03548, 7.88157]
#                [2.01185, 0.84564, 6.16909]
#                [2.79316, 1.71541, 2.97578]
#                [3.22177, 0.16564, 5.79036]
#                [1.81406, 2.74643, 2.13259]
#                [4.77481, 8.01036, 7.57880]]
#        target = np.array([1, 2, 2, 2, 1, 1, 3, 3, 3, 1])
#        dist_metric = lambda x1, x2: np.sum(np.abs(x1-x2), 1)
#
#        # Compute results using method.
#        res = iterative_relief.min_radius(n=n, data=data, target=target, dist_metric=dist_metric, mode='example')
#
#        # Result computed by hand.
#        correct_res = None
#        self.assertEqual(res, correct_res)
#
#    # Test minimal radius for specified number of hypersphere inclusions using distance computations
#    # using indices of examples.
#    def test_min_radius_index(self):
#        iterative_relief = IterativeRelief()
#        
#        # Set parameters
#        n = 3
#        data = [[2.09525, 0.26961, 3.99627]
#                [9.86248, 6.22487, 8.77424]
#                [7.03015, 9.24269, 3.02136]
#                [8.95009, 8.52854, 0.16166]
#                [3.41438, 4.03548, 7.88157]
#                [2.01185, 0.84564, 6.16909]
#                [2.79316, 1.71541, 2.97578]
#                [3.22177, 0.16564, 5.79036]
#                [1.81406, 2.74643, 2.13259]
#                [4.77481, 8.01036, 7.57880]]
#        target = np.array([1, 2, 2, 2, 1, 1, 3, 3, 3, 1])
#        learned_metric_func = partial(lambda data, f, i1, i2: f(data[i1, :], data[i2, :]), data)
#
#        # Compute results using method.
#        res = iterative_relief.min_radius(n=n, data=data, target=target, dist_metric=iterative_relief.dist_func, mode='index', learned_metric_func=learned_metric_func)
#
#        # Result computed by hand.
#        correct_res = None
#        self.assertEqual(res, correct_res)

########################################################


if __name__ == '__main__':
    unittest.main()
