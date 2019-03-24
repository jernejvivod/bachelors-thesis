import pdb
import numpy as np
from PCA_custom import get_dist_func
from relief2 import relief

# Load test data.
data_load = np.loadtxt('rba_test_data2.m')
data = data_load[:, :-1]
target = data_load[:, -1]

# Get learned metric function.
metric_func = get_dist_func(data);


# Apply relief where search for nearest neighbors is done in learned metric space.
rank, weights = relief(data, target, data.shape[0], lambda x1, x2: np.sum(np.abs(x1 - x2)**2, 1)**(1/2), learned_metric_func=metric_func)
print(weights)
