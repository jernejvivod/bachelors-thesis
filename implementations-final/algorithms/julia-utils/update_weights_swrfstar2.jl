function update_weights(data, e, same, other, weights, weights_mult, neigh_weights_same, neigh_weights_other, m, max_f_vals, min_f_vals)

	# Penalty term
	penalty = sum(neigh_weights_same.*(abs.(e .- same)./((max_f_vals .- min_f_vals) .+ 1e-10)), dims=1)

	# Reward term
	reward = sum(neigh_weights_other.*(weights_mult .* (abs.(e .- other)./((max_f_vals .- min_f_vals .+ 1e-10)))), dims=1)

	# Weights update
	weights = weights .- penalty./(m*size(same, 1)) .+ reward./(m*size(other, 1))

	# Return updated weights.
	return vec(weights)
end
