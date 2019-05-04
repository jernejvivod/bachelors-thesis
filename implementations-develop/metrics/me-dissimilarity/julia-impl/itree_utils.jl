using Parameters
export It_node

# structure representing a node in an isolation tree
@with_kw mutable struct It_node
	l
	r
	split_attr::Int
	split_val::Float64
	level::Int
	mass::Int
end
