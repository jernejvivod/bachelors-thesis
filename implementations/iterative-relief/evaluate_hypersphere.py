import numpy as np
 
def evaluate_hypersphere(example, radius, center):
	return np.sum((example-center)**2) <= radius**2