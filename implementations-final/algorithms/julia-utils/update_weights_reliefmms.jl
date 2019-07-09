function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, 
						min_f_vals, dm_vals_same, dm_vals_other, features_msk_same, features_msk_other)

	for t = 1:size(data, 2)

		# Penalty term
		penalty = ((abs.(e[t] .- closest_same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10)) - 
				   dm_vals_same[:,t])[features_msk_same[:, t]]
		penalty = sum(penalty)

		# Reward term
		reward = ((weights_mult .* (abs.(e[t] .- closest_other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10)))) - 
				  dm_vals_other[:,t])[features_msk_other[:, t]]
		reward = sum(reward)

		# Weights update
		weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)
	end

	# Return updated weights.
	return vec(weights)
end
