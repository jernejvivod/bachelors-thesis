function [tree] = get_random_itree(data_sub)
	% Construct and return a random itree.
	%
	% Args:
	% 	data_sub: subset of training examples on which to build the itree.
	% Returns:
	% 	an instance of an In_node class representing the root of the itree
	
	% random_itree: define auxiliary function to implement recursion.
	function [node] = random_itree(x_in, current_height, lim)
		if current_height >= lim || size(x_in, 1) <= 1
			node = It_node(nan, nan, nan, nan, current_height);
		else
			% Randomly select an attribute/dimension.
			q = randi([1, size(x_in, 2)]);

			% Randomly select a split point p between min and max values of attribute q in
			% data subset.
            p = (max(x_in(:, q))-min(x_in(:, q))).*rand() + min(x_in(:, q));

			% Get left and right subtrees.
			xl = x_in(x_in(:, q) < p, :);
			xr = x_in(x_in(:, q) > p, :);
			
			% Recursive case
			node = It_node(random_itree(xl, current_height+1, lim),...
                           random_itree(xr, current_height+1, lim),...
                           q,...
                           p,...
                           current_height);
		end
	end	

	% Build itree.
	tree = random_itree(data_sub, 0, 10);

end
