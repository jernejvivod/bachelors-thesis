function update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals)
	# Update weights 
	weights = weights .- (abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ 1e-10))./m .+
		(abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ 1e-10))./m 
	return weights  # Return updated weights 
end
