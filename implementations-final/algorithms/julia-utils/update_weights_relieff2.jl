function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals)
		
	# Penalty term
	penalty = sum(abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=1)

	# Reward term
	reward = sum(weights_mult .* (abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)

	# Weights update
	weights = weights .- penalty./(m*k) .+ reward./(m*k)

	# Return updated weights.
	return vec(weights)
end
