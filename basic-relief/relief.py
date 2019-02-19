import numpy as np
# input: features and class for each training example, parameter m
# output: vector of feature quality estimations
def relief(data):
	"""
	weights = np.empty(data.shape[1], dtype=float)
	for i in range(m):
		# Randomly choose example R_{i}
		# Find nearest hit H and nearest miss M (closest sample from same class, closest sample from different class)
		# Go over features
		for k, feat in enumerate(data[sample_index, :]):
			# W[A] = W[A] - diff(A, R_{i}, H)/m + diff(A, R_{i}, M)/m
	"""
	return data