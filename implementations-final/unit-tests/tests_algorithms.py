import unittest
from functools import partial
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

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
        res = relief._update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals)

        # Compare with results computed by hand.
        correct_res = np.array([1.004167277552613, 1.0057086828870614, 1.01971232778099])
        assert_array_almost_equal(res, correct_res, decimal=5)

    # Test relief algorithm
    def test_relief(self):

        # training examples
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [7.03015, 9.24269, 3.02136],
                         [4.77481, 8.01036, 7.57880]])

        # class values
        target = np.array([1, 2, 2, 1])

        relief = Relief(n_features_to_select=2, m=data.shape[0])
        relief = relief.fit(data, target)

        # Get results of methods.
        res_rank = relief.rank
        res_weights = relief.weights

        # Compare with results computed by hand.
        correct_res_rank = np.array([1, 2, 3])
        correct_res_weights = np.array([0.11295758, -0.23107757, -0.32095185])

        assert_array_almost_equal(res_rank, correct_res_rank)
        assert_array_almost_equal(res_weights, correct_res_weights)
        

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
        res = relieff._update_weights(data, e[np.newaxis], closest_same, closest_other, weights[np.newaxis], weights_mult[np.newaxis].T, m, k, max_f_vals[np.newaxis], min_f_vals[np.newaxis])

        # Compare with results computed by hand.
        correct_res = np.array([0.01648752,  0.03281824, -0.01643311])
        assert_array_almost_equal(res, correct_res, decimal=5)

    # Test relieff algorithm
    def test_relieff(self):

        # Training examples
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [3.41438, 4.03548, 7.88157],
                         [2.01185, 0.84564, 6.16909],
                         [2.79316, 1.71541, 2.97578],
                         [3.22177, 0.16564, 5.79036],
                         [4.77481, 8.01036, 7.57880]])

        # Class values
        target = np.array([1, 2, 2, 1, 2, 1])

        relieff = Relieff(n_features_to_select=2, k=2, m=data.shape[0])
        relieff = relieff.fit(data, target)

        # Get results of methods.
        res_rank = relieff.rank
        res_weights = relieff.weights

        # Compare with results computed by hand.
        correct_res_rank = np.array([2, 3, 1])
        correct_res_weights = np.array([-0.19887, -0.23507,  0.00803])

        assert_array_equal(res_rank, correct_res_rank)
        assert_array_almost_equal(res_weights, correct_res_weights, decimal=5)

#########################################################



## ITERATIVE RELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ###


from algorithms.iterative_relief import IterativeRelief

class TestIterativeRelief(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        iterative_relief = IterativeRelief()
        self.assertEqual(iterative_relief.n_features_to_select, 10)
        self.assertEqual(iterative_relief.m, -1)
        self.assertEqual(iterative_relief.min_incl, 3)
        self.assertNotEqual(iterative_relief.dist_func, None)
        self.assertEqual(iterative_relief.max_iter, 100)
        self.assertEqual(iterative_relief.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        iterative_relief = IterativeRelief(n_features_to_select=15, m=80, min_incl=5, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(iterative_relief.n_features_to_select, 15)
        self.assertEqual(iterative_relief.m, 80)
        self.assertEqual(iterative_relief.min_incl, 5)
        self.assertNotEqual(iterative_relief.dist_func, None)
        self.assertNotEqual(iterative_relief.learned_metric_func, None)
    
    # Test minimal radius for specified number of hypersphere inclusions using distance computations
    # with explicit examples.
    def test_min_radius_example(self):
        iterative_relief = IterativeRelief()
        
        # Set parameters.
        n = 1
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 3, 1, 3, 2])
        dist_metric = lambda w, x1, x2: np.sum(w*np.abs(x1-x2), 1)

        # Compute results using method.
        res = iterative_relief.min_radius(n=n, data=data, target=target, dist_metric=dist_metric, mode='example')

        # Compare with results computed by hand.
        correct_res = 15.68382
        self.assertAlmostEqual(res, correct_res, places=5)

    # Test minimal radius for specified number of hypersphere inclusions using distance computations
    # using indices of examples.
    def test_min_radius_index(self):
        iterative_relief = IterativeRelief()
        
        # Set parameters
        n = 1
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 3, 1, 3, 2])
        learned_metric_func = partial(lambda data, f, i1, i2: f(data[i1, :], data[i2, :]), data)

        # Compute results using method.
        res = iterative_relief.min_radius(n=n, data=data, target=target, dist_metric=iterative_relief.dist_func, mode='index', learned_metric_func=learned_metric_func)

        # Compare with result computed by hand.
        correct_res = 15.68382
        self.assertAlmostEqual(res, correct_res, places=5)

    # Test iterative Relief algorithm
    def test_iterative_relief(self):
        iterative_relief = IterativeRelief(n_features_to_select=2, min_incl=1, max_iter=1)
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 3, 1, 3, 2])

        # Compute results using method.
        iterative_relief = iterative_relief.fit(data, target)
        res_rank = iterative_relief.rank
        res_weights = iterative_relief.weights

        # Compare with results computed by hand.


########################################################


## I-RELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ########

from algorithms.irelief import IRelief

class TestIRelief(unittest.TestCase):


    # Test initialization with default parameters.
    def test_init_default(self):
        irelief = IRelief()
        self.assertEqual(irelief.n_features_to_select, 10)
        self.assertNotEqual(irelief.dist_func, None)
        self.assertEqual(irelief.max_iter, 100)
        self.assertEqual(irelief.k_width, 5)
        self.assertEqual(irelief.conv_condition, 1.0e-12)
        self.assertEqual(irelief.initial_w_div, 1)
        self.assertEqual(irelief.learned_metric_func, None)


    # Test initialization with explicit parameters.
    def test_init_custom(self):
        irelief = IRelief(n_features_to_select=15, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1),
                max_iter=80, k_width=3, conv_condition=1.0e-15, initial_w_div=2,
                learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(irelief.n_features_to_select, 15)
        self.assertNotEqual(irelief.dist_func, None)
        self.assertEqual(irelief.max_iter, 80)
        self.assertEqual(irelief.k_width, 3)
        self.assertEqual(irelief.conv_condition, 1.0e-15)
        self.assertEqual(irelief.initial_w_div, 2)
        self.assertNotEqual(irelief.learned_metric_func, None)


    def test_pairwise_distances_example(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        irelief = IRelief()

        # Initialize distance function.
        dist_func = lambda w, x1, x2: np.sum(np.abs(w*(x1-x2)))
        dist_weights = np.array([1.2, 0.5, 0.8])
        dist_func_w = partial(dist_func, dist_weights)

        # Compute results using method.
        dist_mat = irelief._get_pairwise_distances(data=data, dist_func=dist_func_w, mode='example')

        # Compare with result computed by hand.
        correct_res = np.array([[0, 16.120682, 2.83908, 3.066782, 2.376784, 9.951871],
                                [16.120682, 0.0, 13.385571, 16.710644, 15.376682, 7.954301],
                                [2.839081, 13.385571, 0.0, 5.905863, 3.540881, 7.21676],
                                [3.066782, 16.710644, 5.905863, 0.0, 2.364982, 10.541833],
                                [2.376784, 15.376682, 3.540881, 2.364982, 0.0, 9.207871],
                                [9.951871, 7.954301, 7.21676, 10.541833, 9.207871, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


    def test_pairwise_distances_index(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        irelief = IRelief()
        dist_weights = np.array([1.2, 0.5, 0.8])

        # Initialize dummy distance function.
        dist_func_learned_dummy = lambda data, w, i1, i2: np.sum(np.abs(w*(data[i1, :]-data[i2, :])))
        dist_func_learned_dummy = partial(dist_func_learned_dummy, data, dist_weights)
        
        # Compute results using method.
        dist_mat = irelief._get_pairwise_distances(data=data, dist_func=dist_func_learned_dummy, mode='index')

        # Compare with result computed by hand.
        correct_res = np.array([[0, 16.120682, 2.83908, 3.066782, 2.376784, 9.951871],
                                [16.120682, 0.0, 13.385571, 16.710644, 15.376682, 7.954301],
                                [2.839081, 13.385571, 0.0, 5.905863, 3.540881, 7.21676],
                                [3.066782, 16.710644, 5.905863, 0.0, 2.364982, 10.541833],
                                [2.376784, 15.376682, 3.540881, 2.364982, 0.0, 9.207871],
                                [9.951871, 7.954301, 7.21676, 10.541833, 9.207871, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


    def test_get_mean_m_vals(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 1, 3, 1])

        # Initialize algorithm.
        irelief = IRelief()

        # Compute results using method.
        mean_m = irelief._get_mean_m_vals(data, target)

        # results computed by hand
        correct_res = np.array([[3.19722, 2.50167667, 2.53085],
                                [6.99316, 3.9321625, 4.60338],
                                [1.12897, 3.0198125, 2.51372  ],
                                [3.47841   , 2.36341667, 3.71420333],
                                [2.231318, 2.9662  , 3.015948],
                                [2.87412   , 5.30838667, 2.52896667]])
        
        # Check for equivalence.
        assert_array_almost_equal(mean_m, correct_res, decimal=5)


    def test_get_mean_h_vals(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 1, 3, 1])

        # Initialize algorithm.
        irelief = IRelief()

        # Compute results using method.
        mean_m = irelief._get_mean_h_vals(data, target)
       
        # results computed by hand
        correct_res = np.array([[0.98691667, 3.40585667, 1.81540333],
                               [3.320355, 3.029615, 1.49194],
                               [3.320355, 3.029615, 1.49194],
                               [1.08064667, 2.58025, 2.43663],
                               [0., 0., 0.],
                               [1.88010333, 4.33489333, 3.00958]])

        # Check for equivalence.
        assert_array_almost_equal(mean_m, correct_res, decimal=5)


    def test_get_gamma_vals(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 1, 3, 1])
        
        # Initialize algorithm.
        irelief = IRelief()
        
        # Initialize distance function.
        dist_func = lambda w, x1, x2: np.sum(np.abs(w*(x1-x2)))
        dist_weights = np.array([1.2, 0.5, 0.8])
        dist_func_w = partial(dist_func, dist_weights)
       
        # Initialize kernel function.
        k_width = 4.0
        kern_func = lambda d: np.exp(-d/k_width)
        
        # Compute pairwise distance matrix.
        dist_mat = irelief._get_pairwise_distances(data, dist_func=dist_func_w, mode='example')

        # Use method to compute gamma values.
        gamma_vals = irelief._get_gamma_vals(dist_mat, target, kern_func)
        
        # results computed by hand
        correct_res = np.array([0.34031707, 0.15538126, 0.02642204, 0.40207921, 0.0, 0.27819147])

        # Check for equivalence.
        assert_array_almost_equal(gamma_vals, correct_res, decimal=5)

    def test_get_nu(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 1, 3, 1])
        
        # Initialize algorithm.
        irelief = IRelief()
        
        # Initialize distance function.
        dist_func = lambda w, x1, x2: np.sum(np.abs(w*(x1-x2)))
        dist_weights = np.array([1.2, 0.5, 0.8])
        dist_func_w = partial(dist_func, dist_weights)
       
        # Initialize kernel function.
        k_width = 4.0
        kern_func = lambda d: np.exp(-d/k_width)
        
        # Compute pairwise distance matrix.
        dist_mat = irelief._get_pairwise_distances(data, dist_func=dist_func_w, mode='example')

        # Use methods to compute gamma values, mean m value and mean h values.
        gamma_vals = irelief._get_gamma_vals(dist_mat, target, kern_func)
        mean_m = irelief._get_mean_m_vals(data, target)
        mean_h = irelief._get_mean_h_vals(data, target)
        
        # Compute nu using method.
        nu = irelief._get_nu(gamma_vals, mean_m, mean_h, data.shape[0])

        # results computed by hand
        correct_res = np.array([0.41760099, 0.00265091, 0.18898647])

        # Check for equivalence.
        assert_array_almost_equal(nu, correct_res, decimal=5)

########################################################


## SURF ALGORITHM IMPLEMENTATION UNIT TESTS ########

from algorithms.SURF import SURF

class TestSURF(unittest.TestCase):


    # Test initialization with default parameters.
    def test_init_default(self):
        surf = SURF()
        self.assertEqual(surf.n_features_to_select, 10)
        self.assertNotEqual(surf.dist_func, None)
        self.assertEqual(surf.learned_metric_func, None)


    # Test initialization with explicit parameters.
    def test_init_custom(self):
        surf = SURF(n_features_to_select=15, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2)), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2)))
        self.assertEqual(surf.n_features_to_select, 15)
        self.assertNotEqual(surf.dist_func, None)
        self.assertNotEqual(surf.learned_metric_func, None)


    # Test computation of pairwise distance matrix using distance function that takes examples.    
    def test_pairwise_distances_example(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        surf = SURF()

        # Initialize distance function.
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))

        # Compute results using method.
        dist_mat = surf._get_pairwise_distances(data=data, dist_func=dist_func, mode='example')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


    # Test computation of pairwise distance matrix using distance function that references examples
    # by their indices.
    def test_pairwise_distances_index(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        surf = SURF()

        # Initialize dummy distance function.
        dist_func_learned_dummy = lambda data, i1, i2: np.sum(np.abs(data[i1, :]-data[i2, :]))
        dist_func_learned_dummy = partial(dist_func_learned_dummy, data)
        
        # Compute results using method.
        dist_mat = surf._get_pairwise_distances(data=data, dist_func=dist_func_learned_dummy, mode='index')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)

    def test_weights_update(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 2, 1, 3, 1])

        # Compute maximal and minimal feature values.
        max_f_vals = np.max(data, 0)
        min_f_vals = np.min(data, 0)

        # Initialize algorithm.
        surf = SURF()

        # Initialize distance function.
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))

        # Compute pairwise distance matrix.
        dist_mat = surf._get_pairwise_distances(data, dist_func=dist_func, mode='example')
        
        # Compute mean distance between examples.
        mean_dist = np.mean(dist_mat)

        # sampled example from dataset
        idx = 2
        e = data[idx, :]
        
        # Get mask for neighbours and exlude distance of example to self from it.
        neigh_mask = dist_mat[idx, :] <= mean_dist
        neigh_mask[idx] = False
        
        # Get masks of hit and miss neighbours.
        hit_neigh_mask = np.logical_and(neigh_mask, target == target[idx])
        miss_neigh_mask = np.logical_and(neigh_mask, target != target[idx])
        
        # Initialize feature weights row vector.
        weights = np.zeros(data.shape[1], dtype=np.float)

        # Update feature weights using method.
        updated_weights = surf._update_weights(data, e[np.newaxis], data[hit_neigh_mask, :], 
                data[miss_neigh_mask, :], weights[np.newaxis], max_f_vals[np.newaxis], min_f_vals[np.newaxis])
                
        # Results computed by hand.
        neigh_mask_correct = np.array([True, False, False, True, True, False])
        hit_neigh_mask_correct = np.array([False, False, False, False, False, False])
        miss_neigh_mask_correct = np.array([True, False, False, True, True, False])
        updated_weights_correct = np.array([0.0204515, 0.02998854, 0.06914647])

        assert_array_equal(neigh_mask, neigh_mask_correct)
        assert_array_equal(hit_neigh_mask, hit_neigh_mask_correct)
        assert_array_equal(miss_neigh_mask, miss_neigh_mask_correct)
        assert_array_almost_equal(updated_weights, updated_weights_correct, decimal=5)

####################################################

## VLSRELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ###

from algorithms.vlsrelief import VLSRelief

class TestVlsRelief(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        vlsrelief = VLSRelief()
        self.assertEqual(vlsrelief.n_features_to_select, 10)
        self.assertEqual(vlsrelief.num_partitions_to_select, 10)
        self.assertEqual(vlsrelief.num_subsets, 10)
        self.assertEqual(vlsrelief.partition_size, 5)
        self.assertEqual(vlsrelief.m, -1)
        self.assertEqual(vlsrelief.k, 5)
        self.assertNotEqual(vlsrelief.dist_func, None)
        self.assertEqual(vlsrelief.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        vlsrelief = VLSRelief(n_features_to_select=15, num_partitions_to_select=15, num_subsets=13, 
                partition_size=3, k=10, m=80, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1),
                learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))

        self.assertEqual(vlsrelief.n_features_to_select, 15)
        self.assertEqual(vlsrelief.num_partitions_to_select, 15)
        self.assertEqual(vlsrelief.num_subsets, 13)
        self.assertEqual(vlsrelief.partition_size, 3)
        self.assertEqual(vlsrelief.k, 10)
        self.assertEqual(vlsrelief.m, 80)
        self.assertNotEqual(vlsrelief.dist_func, None)
        self.assertNotEqual(vlsrelief.learned_metric_func, None)

    # Test partitioning procedure.
    def test_partitioning(sefl):

        # data
        data = np.array([[0.05402045, 0.50067837, 0.88634756, 0.12600894, 0.1217951, 0.79149936, 0.80533926, 0.05799983, 0.09012575, 0.51209482],
                         [0.9710618,  0.87669295, 0.9873201,  0.53634331, 0.7096669, 0.13418176, 0.01828409, 0.74253249, 0.21394251, 0.83794836],
                         [0.2965446,  0.24366306, 0.75521094, 0.31393031, 0.94258869,0.41344823, 0.76089011, 0.74013124, 0.24614211, 0.87928719],
                         [0.44880795, 0.73319887, 0.62020573, 0.67281185, 0.88477685,0.91124333, 0.8338125 , 0.25280089, 0.78730139, 0.14757276],
                         [0.02162485, 0.72887426, 0.67307778, 0.81398892, 0.397657  ,0.80729671, 0.89944621, 0.36769683, 0.90873926, 0.11403073]])
        
        # size of partitions
        partition_size = 3

        # Get array of feature indices.
        feat_ind = np.arange(data.shape[1])
        
        # Test for equality.
        assert_array_equal(feat_ind, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        # Get indices of partition starting indices.
        feat_ind_start_pos = np.arange(0, data.shape[1], partition_size)

        # Test for equality. 
        assert_array_equal(feat_ind_start_pos, np.array([0, 3, 6, 9]))

        # Randomly select k partitions and combine them to form a subset of features of examples.
        ind_sel = np.ravel([np.arange(el, el+partition_size) for el in [9, 0, 6]])

        # Test for equality. 
        assert_array_equal(ind_sel, np.array([9, 10, 11, 0, 1, 2, 6, 7, 8]))
        
        # Remove out of bounds indices.
        ind_sel = ind_sel[ind_sel <= feat_ind[-1]]

        # Test for equality. 
        assert_array_equal(ind_sel, np.array([9, 0, 1, 2, 6, 7, 8]))


####################################################


## BOOSTEDSURF ALGORITHM IMPLEMENTATION UNIT TESTS ###

from algorithms.boostedSURF import BoostedSURF

class TestBoostedSURF(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        boostedsurf = BoostedSURF()
        self.assertEqual(boostedsurf.n_features_to_select, 10)
        self.assertEqual(boostedsurf.phi, 5)
        self.assertNotEqual(boostedsurf.dist_func, None)
        self.assertEqual(boostedsurf.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        boostedsurf = BoostedSURF(n_features_to_select=15, phi=3, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), 
                learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(boostedsurf.n_features_to_select, 15)
        self.assertEqual(boostedsurf.phi, 3)
        self.assertNotEqual(boostedsurf.dist_func, None)
        self.assertNotEqual(boostedsurf.learned_metric_func, None)

    def test_alg(self):

        # data
        data = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 0],
                         [0, 1, 1]])

        # class values
        target = np.array([1, 0, 0, 1, 1, 1])

        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.int)

        # distance function
        dist_func = lambda w, x1, x2: np.sum(w*np.logical_xor(x1, x2), 1)
        dist_weights = np.ones(data.shape[1])
        dist_func_w = partial(dist_func, dist_weights)
        
        # Get next example index.
        idx = 2 

        # Compute distances to all other examples.
        dists = dist_func_w(data[idx, :], data)

        # Compare equality to results computed by hand.
        assert_array_almost_equal(dists, np.array([1., 2., 0., 3., 1., 1.]), decimal=5)


        # Compute mean and standard deviation of distances and set thresholds.
        t_next = np.mean(dists[np.arange(data.shape[0]) != idx])
        sigma_nxt = np.std(dists[np.arange(data.shape[0]) != idx])
        thresh_near = t_next - sigma_nxt/2.0
        thresh_far = t_next + sigma_nxt/2.0


        # Compare equalities to results computed by hand.
        self.assertEqual(t_next, 1.6)
        self.assertEqual(sigma_nxt, 0.8)
        self.assertAlmostEqual(thresh_near, 1.2, places=4)
        self.assertAlmostEqual(thresh_far, 2.0, places=4)

        # Get mask of examples that are close.
        msk_close = dists < thresh_near
        msk_close[idx] = False
        

        # Get mask of examples that are far.
        msk_far = dists > thresh_far


        # Compare equalities to results computed by hand.
        assert_array_equal(msk_close, np.array([True, False, False, False, True, True]))
        assert_array_equal(msk_far, np.array([False, False, False, True, False, False]))


        # Get examples that are close.
        examples_close = data[msk_close, :]
        target_close = target[msk_close]

        # Get examples that are far.
        examples_far = data[msk_far, :]
        target_far = target[msk_far]

        # Get considered features of close examples.
        features_close = data[idx, :] != examples_close
        # Get considered features of far examples.
        features_far = data[idx, :] == examples_far

        # Get mask for close examples with same class.
        msk_same_close = target_close == target[idx]

        # Get mask for far examples with same class.
        msk_same_far = target_far == target[idx]


        ### WEIGHTS UPDATE ###

        # Get penalty weights update values for close examples.
        wu_close_penalty = np.sum(features_close[msk_same_close, :], 0)  # [0, 0, 0]
        # Get reward weights update values for close examples.
        wu_close_reward = np.sum(features_close[np.logical_not(msk_same_close), :], 0)  # [1, 1, 1]

        # Get penalty weights update values for far examples.
        wu_far_penalty = np.sum(features_far[msk_same_far, :], 0)
        # Get reward weights update values for close examples.
        wu_far_reward = np.sum(features_far[np.logical_not(msk_same_far), :], 0)

        # Update weights.
        weights = weights - (wu_close_penalty + wu_far_penalty) + (wu_close_reward + wu_far_reward)

        # Compare equality to results computed by hand.
        assert_array_almost_equal(weights, np.array([1, 1, 1]))



######################################################

## MULTISURF ALGORITHM IMPLEMENTATION UNIT TESTS ###

from algorithms.multiSURF import MultiSURF

class TestMultiSURF(unittest.TestCase):


    # Test initialization with default parameters.
    def test_init_default(self):
        multisurf = MultiSURF()
        self.assertEqual(multisurf.n_features_to_select, 10)
        self.assertNotEqual(multisurf.dist_func, None)
        self.assertEqual(multisurf.learned_metric_func, None)


    # Test initialization with explicit parameters.
    def test_init_custom(self):
        multisurf = MultiSURF(n_features_to_select=15, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(multisurf.n_features_to_select, 15)
        self.assertNotEqual(multisurf.dist_func, None)
        self.assertNotEqual(multisurf.learned_metric_func, None)


    # Test computation of pairwise distance matrix using distance function that takes examples.    
    def test_pairwise_distances_example(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        multisurf = MultiSURF()

        # Initialize distance function.
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))

        # Compute results using method.
        dist_mat = multisurf._get_pairwise_distances(data=data, dist_func=dist_func, mode='example')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


    # Test computation of pairwise distance matrix using distance function that references examples
    # by their indices.
    def test_pairwise_distances_index(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        multisurf = MultiSURF()

        # Initialize dummy distance function.
        dist_func_learned_dummy = lambda data, i1, i2: np.sum(np.abs(data[i1, :]-data[i2, :]))
        dist_func_learned_dummy = partial(dist_func_learned_dummy, data)
        
        # Compute results using method.
        dist_mat = multisurf._get_pairwise_distances(data=data, dist_func=dist_func_learned_dummy, mode='index')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)

    def test_critical_neighbours(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Compute distance matrix.
        multisurf = MultiSURF()
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))
        dist_mat = multisurf._get_pairwise_distances(data=data, dist_func=dist_func, mode='example')
        
        # Compute results using methods.
        idx = 2
        critical_neighbours = multisurf._critical_neighbours(idx, dist_mat)

        # Assert equality to results computed by hand.
        assert_array_equal(critical_neighbours, np.array([0, 4]))

    def test_weights_update(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Class values
        target = np.array([1, 2, 2, 1, 2, 1])
        
        # Initialize feature weights. 
        weights = np.zeros(data.shape[1], dtype=float)



        # Index of next example.
        ex_idx = 2

        # Compute distance matrix.
        multisurf = MultiSURF()
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))
        dist_mat = multisurf._get_pairwise_distances(data=data, dist_func=dist_func, mode='example')

        # Get critical neighbours and mask of critical neighbours with same class.
        r1 = multisurf._critical_neighbours(ex_idx, dist_mat)
        r2 = target[r1] == target[ex_idx] 
        neigh_data = np.vstack((r1, r2))


        # Get classes of miss neighbours.
        classes_other = (target[neigh_data[0]])[np.logical_not(neigh_data[1])]

        # Get probabilities of miss classes.
        u, c = np.unique(classes_other, return_counts=True)
        # Compute weights of classes of miss neighbours.
        class_to_weight = dict(zip(u, c/np.sum(c)))

        # Compute weights multiplier vector.
        weights_mult = np.array([class_to_weight[cls] for cls in classes_other])
        
        # Compute maximum and minimum feature values.
        max_f_vals = np.max(data, 0)
        min_f_vals = np.min(data, 0)

        weights = multisurf._update_weights(data, data[ex_idx, :][np.newaxis], (data[neigh_data[0, :], :])[neigh_data[1, :].astype(np.bool), :],\
                (data[neigh_data[0, :], :])[np.logical_not(neigh_data[1, :].astype(np.bool)), :], weights[np.newaxis], weights_mult[np.newaxis].T, max_f_vals[np.newaxis], min_f_vals[np.newaxis])
        correct_res = np.array([0.01445232, -0.03071705, -0.02560835])

        # Assert equality to results computed by hand.
        assert_array_almost_equal(weights, correct_res)


######################################################


## SURFSTAR ALGORITHM IMPLEMENTATION UNIT TESTS ######

from algorithms.SURFStar import SURFStar

class TestSURFStar(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        surfstar = SURFStar()
        self.assertEqual(surfstar.n_features_to_select, 10)
        self.assertNotEqual(surfstar.dist_func, None)
        self.assertEqual(surfstar.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        surfstar = SURFStar(n_features_to_select=15, 
                dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), 
                learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))

        self.assertEqual(surfstar.n_features_to_select, 15)
        self.assertNotEqual(surfstar.dist_func, None)
        self.assertNotEqual(surfstar.learned_metric_func, None)

    # Test computation of pairwise distance matrix using distance function that takes examples.    
    def test_pairwise_distances_example(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        surfstar = SURFStar()

        # Initialize distance function.
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2))

        # Compute results using method.
        dist_mat = surfstar._get_pairwise_distances(data=data, dist_func=dist_func, mode='example')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


    # Test computation of pairwise distance matrix using distance function that references examples
    # by their indices.
    def test_pairwise_distances_index(self):

        # test data
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])

        # Initialize algorithm.
        surfstar = SURFStar()

        # Initialize dummy distance function.
        dist_func_learned_dummy = lambda data, i1, i2: np.sum(np.abs(data[i1, :]-data[i2, :]))
        dist_func_learned_dummy = partial(dist_func_learned_dummy, data)
        
        # Compute results using method.
        dist_mat = surfstar._get_pairwise_distances(data=data, dist_func=dist_func_learned_dummy, mode='index')

        # Compare with result computed by hand.
        correct_res = np.array([[0.0, 18.50046, 3.02458, 4.62169, 3.1642, 14.00284],
                                [18.50046, 0.0, 15.68382, 18.16851, 17.37724, 8.0686],
                                [3.02458, 15.68382, 0.0, 7.64627, 4.79296, 11.1862],
                                [4.62169, 18.16851, 7.64627, 0.0, 2.85331, 13.6709],
                                [3.1642, 17.37724, 4.79296, 2.85331, 0.0, 12.87962],
                                [14.00284, 8.0686, 11.1862, 13.6709, 12.87962, 0.0]])

        assert_array_almost_equal(dist_mat, correct_res, decimal=5)


######################################################


## MULTISURFSTAR ALGORITHM IMPLEMENTATION UNIT TESTS #

from algorithms.multiSURFStar import MultiSURFStar

class TestMultiSURFStar(unittest.TestCase):

    # Test initialization with default parameters.
    def test_init_default(self):
        multisurfstar = MultiSURFStar()
        self.assertEqual(multisurfstar.n_features_to_select, 10)
        self.assertNotEqual(multisurfstar.dist_func, None)
        self.assertEqual(multisurfstar.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        multisurfstar = MultiSURFStar(n_features_to_select=15, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(multisurfstar.n_features_to_select, 15)
        self.assertNotEqual(multisurfstar.dist_func, None)
        self.assertNotEqual(multisurfstar.learned_metric_func, None)

######################################################


## SWRFSTAR ALGORITHM IMPLEMENTATION UNIT TESTS ######

from algorithms.swrfStar import SWRFStar

class TestSWRFStar(unittest.TestCase):


    # Test initialization with default parameters.
    def test_init_default(self):
        swrfstar = SWRFStar()
        self.assertEqual(swrfstar.n_features_to_select, 10)
        self.assertEqual(swrfstar.m, -1)
        self.assertNotEqual(swrfstar.dist_func, None)
        self.assertEqual(swrfstar.learned_metric_func, None)


    # Test initialization with explicit parameters.
    def test_init_custom(self):
        swrfstar = SWRFStar(n_features_to_select=15, m=80, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(swrfstar.n_features_to_select, 15)
        self.assertEqual(swrfstar.m, 80)
        self.assertNotEqual(swrfstar.dist_func, None)
        self.assertNotEqual(swrfstar.learned_metric_func, None)


    # Test update of feature weights.
    def test_weights_update(self):

        # examples and target values
        data = np.array([[2.09525, 0.26961, 3.99627],
                         [9.86248, 6.22487, 8.77424],
                         [3.22177, 0.16564, 5.79036],
                         [1.81406, 2.74643, 2.13259],
                         [2.79316, 1.71541, 2.97578],
                         [4.77481, 8.01036, 7.57880]])
        target = np.array([1, 2, 3, 1, 3, 2])
        
        # Initialize algorithm.
        swrfstar = SWRFStar()

        
        # distance function
        dist_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1)

        # Initialize feature weights.
        weights = np.zeros(data.shape[1], dtype=np.float)
        
        # Get maximum and minimum feature values.
        max_f_vals = np.max(data, 0)
        min_f_vals = np.min(data, 0)

        # Set m value to number of examples.
        m = data.shape[0]

        # Get all unique classes.
        classes = np.unique(target)

        # Get probabilities of classes in training set.
        p_classes = (np.vstack(np.unique(target, return_counts=True)).T).astype(np.float)
        p_classes[:, 1] = p_classes[:, 1] / np.sum(p_classes[:, 1])


        # index of next sampled example
        idx = 2
        # sampled example
        e = data[idx, :]


        # mask for selecting examples with same class
        # that exludes currently sampled example.
        same_sel = target == target[idx]
        same_sel[idx] = False

        # Get examples with same class and examples with a different class.
        same = data[same_sel, :]
        other = data[target != target[idx], :]
        
        # Get classes of examples with different classes.
        target_other = target[target != target[idx]]

        # Compute distances to examples with same class value.
        distances_same = dist_func(e, same)

        # Compute t and u parameter values for examples with same class value.
        t_same = np.mean(distances_same)
        u_same = np.std(distances_same)

        # Compute distances to examples with a different class.
        distances_other = dist_func(e, other)

        # Compute t and u parameter values for examples with different class value.
        t_other = np.mean(distances_other)
        u_other = np.std(distances_other)

        
        # Compute weights for examples from same class.
        neigh_weights_same = 2.0/(1 + np.exp(-(t_same-distances_same)/(u_same/4.0 + 1e-10)))
        
        # Compute weights for examples from different class.
        neigh_weights_other = 2.0/(1 + np.exp(-(t_other-distances_other)/(u_other/4.0 + 1e-10)))


        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 0] != target[idx], 1]
        
        # Get probabilities of other classes.
        classes_other = p_classes[p_classes[:, 0] != target[idx], 0]


        # Compute diff sum weights for examples from different classes.
        p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == target[idx], 1][0])
        
        # Map weights to 'other' vector and construct weights multiplication vector.
        weights_map = np.vstack((classes_other, p_weights)) 
        weights_mult = np.array([weights_map[1, np.where(weights_map[0, :] == t)[0][0]] for t in target_other])
       

        # Compute updated weights using method.
        weights = swrfstar._update_weights(data, e[np.newaxis], same, other, weights[np.newaxis], weights_mult[np.newaxis].T, 
                neigh_weights_same[np.newaxis].T, neigh_weights_other[np.newaxis].T, m, max_f_vals[np.newaxis], min_f_vals[np.newaxis])
        
        # results computed by hand
        correct_res = np.array([0.00444562, -0.01373921, -0.03862486])
        
        # Assert equality with results computed by hand.
        assert_array_almost_equal(weights, correct_res, decimal=5)

######################################################


## RELIEFSEQ ALGORITHM IMPLEMENTATION UNIT TESTS ######

from algorithms.reliefseq import ReliefSeq

class TestReliefSeq(unittest.TestCase):
    # Test initialization with default parameters.
    def test_init_default(self):
        reliefseq = ReliefSeq()
        self.assertEqual(reliefseq.n_features_to_select, 10)
        self.assertEqual(reliefseq.m, -1)
        self.assertNotEqual(reliefseq.dist_func, None)
        self.assertEqual(reliefseq.learned_metric_func, None)

    # Test initialization with explicit parameters.
    def test_init_custom(self):
        reliefseq = ReliefSeq(n_features_to_select=15, m=80, k_max=10, 
                dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(reliefseq.n_features_to_select, 15)
        self.assertEqual(reliefseq.m, 80)
        self.assertEqual(reliefseq.k_max, 10)
        self.assertNotEqual(reliefseq.dist_func, None)
        self.assertNotEqual(reliefseq.learned_metric_func, None)

    def test_weights_selection(self):
        k_max = 3

        # Initialize matrix of weights by k.
        weights_mat = np.array([[2.09525, 0.26961, 3.99627],
                                [9.86248, 6.22487, 8.77424],
                                [7.03015, 9.24269, 3.02136],
                                [8.95009, 8.52854, 0.16166],
                                [3.41438, 4.03548, 7.88157],
                                [2.01185, 0.84564, 6.16909],
                                [2.79316, 1.71541, 2.97578],
                                [3.22177, 0.16564, 5.79036],
                                [1.81406, 2.74643, 2.13259],
                                [4.77481, 8.01036, 7.57880]])
        
        # Select final weights from weights by k matrix.
        weights = np.max(weights_mat, 1)
        
        # result computed by hand
        correct_res = np.array([3.99627, 9.86248, 9.24269, 8.95009, 7.88157, 6.16909, 2.97578, 5.79036, 2.74643, 8.01036])

        # Assert equality.
        assert_array_almost_equal(weights, correct_res, decimal=5)

#######################################################


if __name__ == '__main__':
    unittest.main()
