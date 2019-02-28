import reliefF
import numpy as np

test_data_large = np.random.rand(260, 17000)
target = ((np.random.rand(260) > 0.5).astype(int))[np.newaxis].T
test_data_large = np.hstack((test_data_large, target))

weights = reliefF.reliefF(test_data_large[:, :-1], test_data_large[:, -1], k=5)