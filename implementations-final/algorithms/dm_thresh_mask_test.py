import numpy as np

e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
c = np.array([2.0, 1.3, 5.5, 3.2, 1.5])

expanded_e = np.tile(e, (e.shape[0], 1))
expanded_c = np.tile(c, (c.shape[0], 1))

# 
np.fill_diagonal(expanded_e, 0)
np.fill_diagonal(expanded_c, 0)

# maximal and minimal feature values.
max_f_vals = np.array([2.0, 3.3, 2.4, 5.4, 5.5])
min_f_vals = np.array([0.1, 0.5, 0.2, 0.1, 0.6])

# Compute DM values and DIFF values.
dm_vals = np.mean(np.abs(expanded_e - expanded_c)/(max_f_vals - min_f_vals), 1)
diff_vals = np.abs(e-c)/(max_f_vals - min_f_vals)

# Get mask of features that are to be considered.
msk = diff_vals > dm_vals


### NOTE ###
# Procedure for computing diff:

# 1. compute weight rewards and penalties.

# 2. set weight rewards and penalties to zero where mask is false.

# Example:

# data
data = np.array([[2.09525, 0.26961, 3.99627],
                 [9.86248, 6.22487, 8.77424],
                 [7.03015, 9.24269, 3.02136],
                 [2.13115, 0.22169, 4.21132],
                 [4.77481, 8.01036, 7.57880]])

# class values
target = np.array([1, 2, 2, 1, 1])

max_f_vals = np.max(data, 0)
min_f_vals = np.min(data, 0)

e = data[0,:]
closest_same = data[-2:,:]
closest_other = data[1:3,:]

# k value
k = 2


diag_msk = np.eye(data.shape[1], dtype=np.bool)
diag_msk_expanded = np.tile(diag_msk, (k, 1))

e_expanded = np.tile(e, (e.size, 1))
np.fill_diagonal(e_expanded, 0.0)
e_expanded = np.tile(e_expanded, (k, 1))

closest_same_expanded = np.repeat(closest_same, closest_same.shape[1], axis=0)
closest_same_expanded[diag_msk_expanded] = 0.0

closest_other_expanded = np.repeat(closest_other, closest_other.shape[1], axis=0)
closest_other_expanded[diag_msk_expanded] = 0.0

dm_vals_same = np.reshape(np.sum(np.abs(e_expanded - closest_same_expanded)/(max_f_vals - min_f_vals), 1)/np.float(data.shape[1]-1), (k, data.shape[1]))
diff_vals_same = np.abs(e - closest_same)/(max_f_vals - min_f_vals)

dm_vals_other = np.reshape(np.sum(np.abs(e_expanded - closest_other_expanded)/(max_f_vals - min_f_vals), 1)/np.float(data.shape[1]-1), (k, data.shape[1]))
diff_vals_other = np.abs(e - closest_other)/(max_f_vals - min_f_vals)

features_msk_same = diff_vals_same > dm_vals_same
features_msk_other = diff_vals_other > dm_vals_other


