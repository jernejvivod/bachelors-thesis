import unittest
from algorithms.relief import Relief

## RELIEF ALGORITHM IMPLEMENTATION UNIT TESTS ##########

class TestRelief(unittest.TestCase):
    def test_init_default(self):
        relief = Relief()
        self.assertEqual(relief.n_features_to_select, 10)
        self.assertEqual(relief.m, 100)
        self.assertNotEqual(relief.dist_func, None)
        self.assertEqual(relief.learned_metric_func, None)


    def test_init_custom(self):
        relief = Relief(n_features_to_select=15, m=80, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(relief.n_features_to_select, 15)
        self.assertEqual(relief.m, 80)
        self.assertNotEqual(relief.dist_func, None)
        self.assertNotEqual(relief.learned_metric_func, None)

    def test_weights_update(self):
        relief = Relief()

        # Initialize parameter values.
        data = None
        e = None
        closest_same = None
        closest_other = None
        weights = None
        m = None
        max_f_vals = None
        min_f_vals = None
        
        # Compute weights update
        res = relief._update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals) 

        # Result computed by hand.
        correct_res = None
        self.assertAlmostEqual(res, correct_res, places=3)

########################################################





## RELIEFF ALGORITHM IMPLEMENTATION UNIT TESTS #########

from algorithms.relieff import Relieff

class TestRelieff(unittest.TestCase):
    def test_init_default(self):
        relieff = Relieff()
        self.assertEqual(relieff.n_features_to_select, 10)
        self.assertEqual(relieff.m, 100)
        self.assertEqual(relieff.k, 5)
        self.assertNotEqual(relieff.dist_func, None)
        self.assertEqual(relieff.learned_metric_func, None)


    def test_init_custom(self):
        relieff = Relieff(n_features_to_select=15, m=80, k=3, dist_func=lambda x1, x2: np.sum(np.abs(x1-x2), 1), learned_metric_func = lambda x1, x2: np.sum(np.abs(x1-x2), 1))
        self.assertEqual(relieff.n_features_to_select, 15)
        self.assertEqual(relieff.m, 80)
        self.assertEqual(relieff.k, 3)
        self.assertNotEqual(relieff.dist_func, None)
        self.assertNotEqual(relieff.learned_metric_func, None)

    def test_weights_update(self):
        relieff = Relieff()

        # Initialize parameter values.
        data = None
        e = None
        closest_same = None
        closest_other = None
        weights = None
        weights_mult = None
        m = None
        k = None
        max_f_vals = None
        min_f_vals = None
        
        # Compute weights update
        res = relieff._update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals)

        # Result computed by hand.
        correct_res = None
        self.assertAlmostEqual(res, correct_res, places=3)

########################################################



if __name__ == '__main__':
    unittest.main()
