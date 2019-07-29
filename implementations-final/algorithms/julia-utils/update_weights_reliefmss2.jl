function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, 
						min_f_vals, dm_vals_same, dm_vals_other, features_msk_same, features_msk_other)

	# Penalty term
	penalty = (abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ 1e-10)) - dm_vals_same
	penalty[.!features_msk_same] .= 0.0
	penalty = sum(penalty, dims=1)

	# Reward term
	reward = weights_mult .* (abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ 1e-10)) - dm_vals_other
	reward[.!features_msk_other] .= 0.0
	reward = sum(reward, dims=1)

	# Weights update
	weights = weights .- penalty./(m*k) .+ reward./(m*k)

	# Return updated weights.
	return vec(weights)
end
