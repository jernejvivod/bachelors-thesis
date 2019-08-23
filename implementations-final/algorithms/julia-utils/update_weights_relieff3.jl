function update_weights(data, e, closest_same, closest_other, weights, weights_mult, m, k, max_f_vals, min_f_vals)

	# Distance weights for penalty term
	
	# Compute weights for each nearest hit
	d_vals_closest_same = 1 ./ (sum(abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
	dist_weights_penalty = d_vals_closest_same ./ sum(d_vals_closest_same)

	# Distance weights for reward term
	d_vals_closest_other = 1 ./ (sum(abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
	dist_weights_reward = d_vals_closest_other ./ sum(d_vals_closest_other)
		
	# Penalty term
	penalty = sum(dist_weights_penalty .* (abs.(e .- closest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
	
	# Reward term
	reward = sum(weights_mult .* (dist_weights_reward .* (abs.(e .- closest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

	# Weights update
	weights = weights .- penalty./m .+ reward./m

	# Return updated weights.
	return vec(weights)
end
