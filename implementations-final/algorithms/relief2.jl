using StatsBase

function relief(data, target, m, dist_func)

    # Initialize feature weights vector.
    weights = zeros(Float16, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = vec(maximum(data, dims=1))
    min_f_vals = vec(minimum(data, dims=1))
    
    # Sample m examples without replacement.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end

    # Go over sampled indices.
    for idx = sample_idxs

        # Find nearest miss and neareset hit.
        nearest_hit = data[vec((target .== target[idx]) .& (1:size(data,1) .!= idx)), :][argmin(dist_func(data[idx:idx, :], data[vec((target .== target[idx]) .& (1:size(data,1) .!= idx)), :]))[1], :]
        nearest_miss = data[vec(target .!= target[idx]), :][argmin(dist_func(data[idx:idx, :], data[vec(target .!= target[idx]), :]))[1], :]
        
        ### Weights Update ###
        
        weights = weights .- (abs.(data[idx, :] .- nearest_hit)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m .+
            (abs.(data[idx, :] .- nearest_miss)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m 

        ######################
    end

    # Return computed feature weights.
    return weights
end

