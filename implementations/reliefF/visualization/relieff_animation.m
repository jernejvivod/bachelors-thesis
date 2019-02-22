function [weights] = relieff_animation(data, m, k, dist_func, timeout, use_deletions)
	weights = zeros(1, size(data, 2) - 1);  % Initialize weights.
	idx_sampled = randsample(1:size(data, 1), m);
	
	classes = unique(data(:, end));
	for idx = idx_sampled
		
		e = data(idx, :);
		
		% Get index of sampled example in group of examples of same class.
		data_class_aux = data(1:idx-1, end); idx_class = idx - sum(data_class_aux ~= e(end));
		
		% Find k nearest examples from same class.
		distances_same = dist_func(repmat(e(1:end-1), sum(data(:, end) == e(end)), 1), data(data(:, end) == e(end), 1:end-1));
		distances_same(idx_class) = inf;
		[~, idxs_closest_same] = mink(distances_same, k);
		data_same_aux = data(data(:, end) == e(end), :);
		closest_same = data_same_aux(idxs_closest_same, :);
		
		
		% Can remove leading class column as order follows the classes
		% vector.
		
		classes_vect = repmat(classes(classes ~= e(end)), 1, k)'; classes_vect = classes_vect(:);
		closest_other = [classes_vect, zeros(k * (length(classes) - 1), size(data, 2))];
		top_ptr = 1;
		for cl = classes'
			if cl ~= e(end)
				distances_cl = dist_func(repmat(e(1:end-1),  sum(data(:, end) == cl), 1), data(data(:, end) == cl, 1:end-1));
				[~, idx_closest_cl] = mink(distances_cl, k);
				data_cl_aux = data(data(:, end) == cl, :);
				closest_other(top_ptr:top_ptr+k-1, 2:end) = data_cl_aux(idx_closest_cl, :);
				top_ptr = top_ptr + k;
			end
		end
		
		% WORKS TO HERE
		
		for k = 1:size(data, 2) - 1
			% sum1 = 
			% sum2 =
			% sum3 =
			% weights(k) = 
		end
		
	end
end