function update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals)
	for t = 1:size(data, 2)
		# Update weights 
		weights[t] = weights[t] - (abs(e[t] - closest_same[t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))/m +
			(abs(e[t] - closest_other[t])/((max_f_vals[t] - min_f_vals[t]) + 1e-10))/m 
	end
	return weights  # Return updated weights 
end
