function update_weights(data, e, closest_same, closest_other, weights, m, max_f_vals, min_f_vals)
	weights = weights .- (abs.(e .- closest_same)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m .+
		(abs.(e .- closest_other)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m 
	return weights
end
