function [] = get_node_masses(data, random_itrees)
    % Set mass property of each node by traversing trees with each instance
	% in training set.
	%
    % Args:
    %     data: training data
    %     random_itrees: list of random itrees to traverse and set mass property
    % Returns:
    %     list of random itrees with set mass property

	% traverse: traverse tree with example and increment
	% masses of visited nodes
	function [] = traverse(example, it_node)
		if it_node.l == nan && it_node.r == nan
			it_node.mass = it_node.mass + 1;
		elseif example[it_node.split_attr] < it_node.split_val
			it_node.mass = it_node.mass + 1;
			traverse(example, it_node.l);
		else
			it_node.mass = it_node.mass + 1;
			traverse(example, it_node.r);
		end
	end

	% compute_masses: compute mass of itree
	% using passed training data.
	function [] = compute_mass(data, itree)
		for example = data
			traverse(example, itree);
		end
	end

	% Go over itrees in list and compute masses.
	for itree = random_itrees
		compute_mass(data, itree);
	end

end
