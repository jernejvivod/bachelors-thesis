import numpy as np
import relieff
import reliefF

def minkowski_distance(e1, e2, p):
	return np.sum(abs(e1 - e2)**p, 1)**(1/p)

test_data = np.loadtxt('rba_test_data2.m')

test_data_large = np.random.rand(200, 17000)
target = ((np.random.rand(200) > 0.5).astype(int))[np.newaxis].T
test_data_large = np.hstack((test_data_large, target))

ranks1, weights1 = relieff.relieff(test_data, test_data.shape[0], 5, lambda a, b: minkowski_distance(a, b, 2));
# weights2 = reliefF.reliefF(test_data[:, :-1], test_data[:, -1],  k=5);