using StatsBase

function relief(data, target, m, dist_func)

    # Initialize feature weights vector.
    weights = zeros(Float16, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Sample m examples without replacement.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)

    # Go over sampled indices.
    for idx = sample_idxs
        
        # Find distances of sampled sample to other samples with same or different class values.
        dist_same = dist_func(data[idx:idx, :], data[vec(target .== target[idx]), :])
        dist_other = dist_func(data[idx:idx, :], data[vec(target .!= target[idx]), :])

        # Find indices of nearest hits and misses within samples with same or different class values.
        nearest_hit_idx = argmin(dist_same)
        nearest_miss_idx = argmin(dist_other)
        
        # Get nearest hits and nearest misses.
        nearest_hit = data[vec(target .== target[idx]), :][nearest_hit_idx[1], :]
        nearest_miss = data[vec(target .!= target[idx]), :][nearest_miss_idx[1], :]

        ### Update weights ###
        
        println(nearest_hit)
        println(nearest_miss)
        

        ######################

    end

end

data = [1 2 3; 4 5 6; 7 8 9]
target = [0, 1, 0]
dist_func = function(e1, e2) return sum(abs.(e1.-e2), dims=2) end

relief(data, target, -1, dist_func)

