import numpy as np
import pdb

def relieff(data, m, k, dist_func):


	def diff(idx_feature, e1, e2, max_f_val, min_f_val):
		if True:
			return abs(e1[:, idx_feature] - e2[:, idx_feature])/(max_f_val - min_f_val)
		else:
			return 0 if e1[idx_feature] == e2[idx_feature] else 1

	def movsum(l, k):
		final_arr = np.zeros(len(l))
		for m in np.arange(k):
			final_arr = final_arr + np.roll(l, -m)
		return final_arr

	weights = np.zeros(data.shape[1] - 1, dtype=float) #
	idx_sampled = np.random.choice(np.arange(data.shape[0]), m, replace=False) #

	max_f_vals = np.amax(data[:, :-1], 0) #
	min_f_vals = np.amin(data[:, :-1], 0) #

	classes = np.unique(data[:, -1])
	p_classes = np.vstack(np.unique(data[:, -1], return_counts=True)).T #
	p_classes[:, 1] = p_classes[:, 1] / np.sum(p_classes[:, 1]) #

	for idx in idx_sampled:
		e = data[idx, :] #
		idx_class = idx - np.sum(data[:idx, -1] != e[-1]) #
		
		# Find k nearest examples from same class.
		distances_same = dist_func(np.tile(e[:-1], (np.sum(data[:, -1] == e[-1]), 1)), data[data[:, -1] == e[-1], :-1]) #
		distances_same[idx_class] = np.inf #

		idxs_closest_same = np.argpartition(distances_same, k)[:k] #

		closest_same = (data[data[:, -1] == e[-1], :])[idxs_closest_same, :] #

		# Allocate matrix template for getting nearest examples from other classes.
		closest_other = np.zeros((k * (len(classes) - 1), data.shape[1])) #
		top_ptr = 0 # # Initialize pointer for adding examples to template matrix.
		for cl in classes:  # Go over classes different than the one of current sampled example.
			if cl != e[-1]:
				# Get closest k examples with class cl
				distances_cl = dist_func(np.tile(e[:-1],  (np.sum(data[:, -1] == cl), 1)), data[data[:, -1] == cl, :-1]) #
				idx_closest_cl = np.argpartition(distances_cl, k)[:k] #
				# Add found closest examples to matrix.
				closest_other[top_ptr:top_ptr+k, :] = (data[data[:, -1] == cl, :])[idx_closest_cl, :] #
				top_ptr = top_ptr + k #


		# Get probabilities of classes not equal to class of sampled example.
		p_classes_other = p_classes[p_classes[:, 0] != e[-1], 1] #
		

		# Compute diff sum weights for closest examples from different class.
		p_weights = p_classes_other/(1 - p_classes[p_classes[:, 0] == e[-1], 1]) #
		
		# Go over features.
		for t in np.arange(data.shape[1]-1):
			# Update feature weights.
			sum1 = np.sum(diff(t, np.tile(e[:-1], (k, 1)), closest_same[:, :-1], max_f_vals[t], min_f_vals[t])/(m*k))
			sum2 = movsum(diff(t, np.tile(e[:-1], (k*(len(classes)-1), 1)), closest_other[:, :-1], max_f_vals[t], min_f_vals[t]), k)
			sum2 = sum2[0:len(sum2):k]
			sum3 = np.sum(p_weights * sum2)
			weights[t] = weights[t] - sum1 + sum3/(m*k)

	return weights
