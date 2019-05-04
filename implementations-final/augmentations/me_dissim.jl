using Pkg, Random, Distributions, StatsBase, Parameters
using Pkg
ENV["PYTHON"] = "python3.7"
Pkg.build("PyCall")

Random.seed!(123)

# structure representing a node in an isolation tree
@with_kw mutable struct It_node
	l::Union{It_node, Nothing}
	r::Union{It_node, Nothing}
	split_attr::Union{Int, Nothing}
	split_val::Union{Float64, Nothing}
	level::Int
	mass::Int = 0 
end


# !!!TESTED!!!
function get_random_itree(data_sub)

    # random_itree: define auxiliary function to implement recursion.
    function random_itree(x_in; current_height, lim)
		if current_height >= lim || size(x_in, 1) <= 1 # Base case check
        	return It_node(l=nothing, r=nothing, split_attr=nothing, split_val=nothing, level=current_height)
        else
        # Randomly select an attribute q.
        q = rand(1:size(x_in, 2))
        # Randomly select a split point p between min and max values of attribute q in X.
		min_q_x_in = minimum(x_in[:, q])
		max_q_x_in = maximum(x_in[:, q])
		if min_q_x_in == max_q_x_in
			p = min_q_x_in
		else
        	p = rand(Uniform(minimum(x_in[:, q]), maximum(x_in[:, q])))
		end
        # Get left and right subtrees.
        xl = x_in[x_in[:, q] .< p, :]
        xr = x_in[x_in[:, q] .>= p, :]
        # Recursive case
        return It_node(l=random_itree(xl, current_height=current_height+1, lim=lim),
                   r=random_itree(xr, current_height=current_height+1, lim=lim),
                   split_attr=q,
                   split_val=p,
                   level=current_height)
        end
    end

    # Build itree
    return random_itree(data_sub, current_height=0, lim=10)
end


# !!!TESTED!!!
function get_n_random_itrees(n, subs_size, data)
    random_itrees = Array{It_node, 1}(undef, n)
    # TODO: parallelize!
    for k = 1:n
        # Get a random sample of training examples to build next random itree.
        data_sub = data[sample(1:size(data, 1), subs_size, replace=false), :]
        random_itrees[k] = get_random_itree(data_sub)  # Get next random itree
    end
    return random_itrees, subs_size
end


# !!!TESTED!!!
function get_lowest_common_node_mass(itree, x1, x2)

    # If node is a leaf, return its mass.
    if itree.split_val == nothing
    	return itree.mass
    end

    # If x1 and x2 are in different subtrees, return current node's mass.
	if (x1[itree.split_attr] < itree.split_val) != (x2[itree.split_attr] < itree.split_val)
    	return itree.mass
    end

    # If both examples if left subtree, make recursive call.
	if (x1[itree.split_attr] < itree.split_val) && (x2[itree.split_attr] < itree.split_val)
    	return get_lowest_common_node_mass(itree.l, x1, x2)
    end

    # If both examples in right subtree, make recursive call.
	if (x1[itree.split_attr] >= itree.split_val) && (x2[itree.split_attr] >= itree.split_val)
    	return get_lowest_common_node_mass(itree.r, x1, x2)
    end
end


# !!!TESTED!!!
function mass_based_dissimilarity(x1, x2, itrees, subs_size)
    # In each i-tree, find lowest nodes containing both examples and accumulate masses.
    sum_masses = 0
    for i = 1:length(itrees)
    	sum_masses += get_lowest_common_node_mass(itrees[i], x1, x2)/subs_size
    end

    return (1/length(itrees)) * sum_masses  # Divide by number of space partitioning models.

end


# !!!TESTED!!!
function get_node_masses(itrees, data)

    # traverse: traverse itree with example and increment masses of visited nodes
    function traverse(example, it_node)

        # base case - in leaf
        if it_node.l == nothing && it_node.r == nothing
        	it_node.mass += 1

        # if split attribute value lower than split value
        elseif example[it_node.split_attr] < it_node.split_val
        	it_node.mass += 1
        	traverse(example, it_node.l)  # Traverse left subtree.

        # if split attribute value greater or equal to split value
        else
        	it_node.mass += 1
        	traverse(example, it_node.r)  # Traverse right subtree.
        end
    end


    # compute_masses: compute masses of nodes in itree
    function compute_masses(itree, data)
		for example_idx in 1:size(data, 1)
			traverse(data[example_idx, :], itree)
        end
    end


    # TODO: parallelize!
    for itree in itrees  # Go over itrees and set masses of nodes.
        compute_masses(itree, data)
    end
end


function get_dissim_func(num_itrees, data)
    itrees, subs_size = get_n_random_itrees(num_itrees, size(data, 1), data)
    get_node_masses(itrees, data)
	res_func = (i1, i2) -> mass_based_dissimilarity(data[i1.+1, :], data[i2.+1, :], itrees, subs_size)
	return (i1, i2, kwargs...) -> res_func.(i1, i2)
end

