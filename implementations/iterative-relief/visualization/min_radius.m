function [r] = min_radius(n, data, target) 
	% function [r] = min_radius(n, data, target) 
	%
	% Compute minimum radius of a hypersphere such that a hypersphere with
	% such radius centered at every example in the data matrix contains at
	% least n other examples from same and n examples from another class.
	%
	% n --- minimum number of examples from same class and examples from
	% other classes each hypersphere should contain
	% data --- matrix of training data features in rows
	% target --- matrix containing the class values for each example
	%
	% Author: Jernej Vivod
	
	% Allocate array for storing minimum radii corresponding to each example.
	min_r = zeros(size(data, 1), 1);
	
	% Compute a pairwise distance matrix using Euclidean distance.
	dist_mat = squareform(pdist(data, 'minkowski', 2));
	
	% Go over examples.
	for k = 1:size(data, 1)
		% Get corresponding row in distance matrix.
		dist_from_e = dist_mat(k, :);
		% Get mask for examples from same class.
		msk = target == target(k);
		% Get distances to examples from same class.
		dist_same = dist_from_e(msk);
		% Get distances to examples from a different class.
		dist_diff = dist_from_e(~msk);
		
		% Sort vectors of distances.
		min_list1 = sort(dist_same); min_list2 = sort(dist_diff);
		% Ger minimum acceptable radius for hypersphere centered at this example.
		min_r(k) = max(min_list1(n+1), min_list2(n)); % (Handle distance of example to itself)
	end
	
	% Result is the maximum value in the vector of minimum radius sizes.
	r = max(min_r);
	
end