function [weights] = relief_animation(data, m, dist_func)
	% Create an animation of the basic Relief feature selection algorithm
	% using three dimensional feature space.
	%
	% data --- matrix of training data with class column as the last column.
	% m --- the m parameter (example sample size)
	% dist_func --- distance function for evaluating distance between
	% examples. The function should be able to take two matrices of
	% examples and return a vector of distances between the examples.

	% diff: score distance between two examples
	function [res] = diff(idx_feature, e1, e2, max_f_val, min_f_val)
		% if feature continuous
		% TODO continuous? optional argument.
		if 1
			res = abs(e1(idx_feature) - e2(idx_feature))/(max_f_val - min_f_val);
		else  % if feature discrete
			if e1(idx_feature) == e2(idx_feature); res = 0; else; res = 1; end
		end
	end

	% Initialize all weights to 0.
	weights = zeros(1, size(data, 2) - 1);
	
	% Sample m examples.
	idx_sampled = randsample(1:size(data, 1), m);
	
	% Compute minimum and maximum feature values.
	max_f_vals = max(data(:, 1:end-1));
	min_f_vals = min(data(:, 1:end-1));

	
	% Create scatter plot of data
	
	
	
	for idx = idx_sampled
		
		% Display current weight values.
		
		e = data(idx, :);  % Get example that was sampled.
		
		
		% Mark selected example.
		
		
		
		% Get index of sampled example in subset of examples with same class.
		data_class_aux = data(1:idx-1, end); idx_class = idx - sum(data_class_aux ~= e(end));  
		
		% Find nearest example from same class (H) and nearest example from different class (M).
		distances_same = dist_func(repmat(e(1:end-1), sum(data(:, end) == e(end)), 1), data(data(:, end) == e(end), 1:end-1));
		distances_diff = dist_func(repmat(e(1:end-1), sum(data(:, end) ~= e(end)), 1), data(data(:, end) ~= e(end), 1:end-1));
		distances_same(idx_class) = inf;
		
		% Get indices of examples with critical distances.
		[~, idx_closest_same] = min(distances_same);
		[~, idx_closest_diff] = min(distances_diff);
		
		% Get closest example from same class and closest example from different class.
		data_same = data(data(:, end) == e(end), :); closest_same = data_same(idx_closest_same, :);
		data_diff = data(data(:, end) ~= e(end), :); closest_diff = data_diff(idx_closest_diff, :);
		
		
		% Plot distances to H and M using different coloured lines and
		% display distances on screen.
		
		% Display message: 'Scoring features'.
		
		% Go over features
		for k = 1:size(data, 2)-1
			weights(k) = weights(k) - diff(k, e, closest_same, max_f_vals(k), min_f_vals(k))/m + diff(k, e, closest_diff, max_f_vals(k), min_f_vals(k))/m;
		end
		
	end
end