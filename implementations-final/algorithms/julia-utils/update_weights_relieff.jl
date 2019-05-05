function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals)

	for t = 1:size(data, 2)

		# Penalty term
		penalty = sum(abs.(e[t] .- closest_same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10))

		# Reward term
		reward = sum(weights_mult .* (abs.(e[t] .- closest_other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10))))

		# Weights update
		weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)
	end

	# Return updated weights.
	return weights
end
