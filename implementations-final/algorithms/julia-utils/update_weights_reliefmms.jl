function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals)

	# Compute scaled diff values for hits and misses.
	dmh = (1/(size(data, 2)-1)).*(abs.(e .- closest_same)./(max_f_vals .- min_f_vals))
	dmm = (1/(size(data, 2)-1)).*(abs.(e .- closest_other)./(max_f_vals .- min_f_vals))

	for t = 1:size(data, 2)
		# Compute DMH and DMM values 
		dmh_temp = copy(dmh)
		dmh_temp[:, t] .= 0
		dmh_nxt = sum(dmh_temp[t, :], dims=2)

		dmm_temp = copy(dmm)
		dmm_temp[:, t] = 0
		dmm_nxt = sum(dmm_temp[t, :], dims=2)

		# TODO diffH > DMH and diffM > DMM

		# Penalty term
		penalty = sum((abs.(e[t] .- closest_same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10)) - dmh)

		# Reward term
		reward = sum((weights_mult .* (abs.(e[t] .- closest_other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10)))) - dmm)

		# Weights update
		weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)
	end

	# Return updated weights.
	return weights
end

