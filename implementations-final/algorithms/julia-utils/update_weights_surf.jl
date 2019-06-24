function update_weights(data, e, closest_same, closest_other, weights, max_f_vals, min_f_vals)

	for t = 1:size(data, 2)

		# Penalty term
		penalty = sum(abs.(e[t] .- closest_same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10))

		# Reward term
		reward = sum(abs.(e[t] .- closest_other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10)))

		# Weights update
		weights[t] = weights[t] - penalty/(size(data, 1)*size(closest_same, 1) + 1e-10) + reward/(size(data, 1)*size(closest_other, 1) + 1e-10)
	end

	# Return updated weights.
	return weights
end
