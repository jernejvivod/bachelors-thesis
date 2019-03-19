function [random_trees] = get_n_random_itrees(n, data, sub_size)
	% Construct and return n random itrees in a list
	%
	% Args:
	% 	n: number of random itrees to construct
	% 	data: data on which to build the itree
	% 	sub_size: size of the subset of examples used to build the tree
	%
	random_trees = cell(n);  % Define list for storing trees.

	for k = 1:n
        % Get data subset and construct next tree.
		data_sub = data(randsample(1:size(data, 1), sub_size), :);
		random_trees{k} = get_random_itree(data_sub);
	end
end
