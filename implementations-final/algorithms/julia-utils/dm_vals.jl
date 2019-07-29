import Statistics

function dm_vals(e, closest, max_f_vals, min_f_vals)
	results =  Array{Float64, 2}(undef, size(closest))
	diff_vals = abs.(e .- closest)./(max_f_vals .- min_f_vals .+ eps(Float64))
	for i = 1:length(e)
		results[:,i] = Statistics.mean(diff_vals[:, 1:end .!= i], dims=2)
	end

	return results
end
