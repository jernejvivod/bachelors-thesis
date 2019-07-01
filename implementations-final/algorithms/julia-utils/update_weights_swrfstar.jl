function update_weights(data, e, same, other, weights, weights_mult, neigh_weights_same, neigh_weights_other, m, max_f_vals, min_f_vals)

	for t = 1:size(data, 2)

		# Penalty term
		penalty = sum(neigh_weights_same.*(abs.(e[t] .- same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10)))

		# Reward term
		reward = sum(neigh_weights_other.*(weights_mult .* (abs.(e[t] .- other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10)))))

		# Weights update
		weights[t] = weights[t] - penalty/(m*size(same, 1)) + reward/(m*size(other, 1))
	end

	# Return updated weights.
	return weights
end
