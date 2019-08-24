function update_weights(data, e, closest_same, closest_other, weights, weights_mult, closest_same_weights, closest_other_weights, m, k, max_f_vals, min_f_vals)

	# Penalty term
	penalty = sum(closest_same_weights .* (abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
	
	# Reward term
	reward = sum(weights_mult .* (closest_other_weights .* (abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

	# Weights update
	weights = weights .- penalty./m .+ reward./m

	# Return updated weights.
	return vec(weights)
end
