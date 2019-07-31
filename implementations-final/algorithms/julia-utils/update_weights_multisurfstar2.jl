function update_weights(data, e, closest_same, closest_other, weights, weights_mult, max_f_vals, min_f_vals)

	# Penalty term
	penalty = sum(abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ 1e-10), dims=1)

	# Reward term
	reward = sum(weights_mult .* abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ 1e-10), dims=1)

	# Weights update
	weights = weights .- penalty./(size(data, 1)*size(closest_same, 1) + 1e-10) .+ reward./(size(data, 1)*size(closest_other, 1) + 1e-10)

	# Return updated weights.
	return vec(weights)
end
