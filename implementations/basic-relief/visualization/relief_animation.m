function [rank, weights] = relief_animation(data, m, dist_func, plot, timeout, use_deletions)
	% function [weights] = relief_animation(data, m, dist_func, timeout, use_deletions)
	%
	% Create an animation of the basic Relief feature selection algorithm
	% using three dimensional feature space.
	%
	% data --- matrix of training data with class column as the last column.
	% m --- the m parameter (example sample size)
	% dist_func --- distance function for evaluating distance between
	% examples. The function should be able to take two matrices of
	% examples and return a vector of distances between the examples.
	% plot --- if set to 0, the function only computes the weights with no
	% plotting.
	% timeout --- duration of pause between graphical object plotting function calls.
	% use_deletions --- logical value that specifies whether to delete the
	% computation visualization when moving onto next example in the
	% sample.
	%
	% Author: Jernej Vivod
	
	% Set parameter values if not present in call.
	if nargin < 6
		use_deletions = 1;
		if nargin < 5
			timeout = 0.3;
			if nargin < 4
				plot = 0;
			end
		end
	end

	% Initialize all weights to 0.
	weights = zeros(1, size(data, 2) - 1);
	
	% Sample m examples.
	idx_sampled = randsample(1:size(data, 1), m);
	
	% Compute minimum and maximum feature values.
	max_f_vals = max(data(:, 1:end-1));
	min_f_vals = min(data(:, 1:end-1));

	
	
	% ### PLOTTING ###
	if plot
		% Create scatter plot of data
		figure(1); hold on;
		scatter3(data(:, 1), data(:, 2), data(:, 3), 30, categorical(data(:, 4)), 'filled');
		xlabel('a'); ylabel('b'); zlabel('c'); view(30, 50); grid on;
		pause on
	end
	% ### /PLOTTING ###
	
	
	
	for idx = idx_sampled
		
	
		
		% ### PLOTTING ###
		if plot
			% Display current weight values.
			hT = title({'Relief Algorithm Visualization', sprintf('$$ weights = [%.3f, %.3f, %.3f] $$', weights(1), weights(2), weights(3))},'interpreter','latex');
			set(hT, 'FontSize', 17);
		end
		% ### /PLOTTING ###
		
		
		
		e = data(idx, :);  % Get example that was sampled.
		
		
		
		% ### PLOTTING ###
		if plot
			% Mark selected example.
			sample_p = plot3(e(1), e(2), e(3), 'ro', 'MarkerSize', 10);
			pause(timeout);
		end
		% ### /PLOTTING ###
		
		
		
		% Get index of sampled example in subset of examples with same class.
		data_class_aux = data(1:idx-1, end); idx_class = idx - sum(data_class_aux ~= e(end));
		
		% Find nearest example from same class (H) and nearest example from differensample_pt class (M).
		distances_same = dist_func(repmat(e(1:end-1), sum(data(:, end) == e(end)), 1), data(data(:, end) == e(end), 1:end-1));
		distances_diff = dist_func(repmat(e(1:end-1), sum(data(:, end) ~= e(end)), 1), data(data(:, end) ~= e(end), 1:end-1));
		distances_same(idx_class) = inf;
		
		% Get indices of examples with critical distances.
		[~, idx_closest_same] = min(distances_same);
		[~, idx_closest_diff] = min(distances_diff);
		
		% Get closest example from same class and closest example from different class.
		data_same = data(data(:, end) == e(end), :); closest_same = data_same(idx_closest_same, :);
		data_diff = data(data(:, end) ~= e(end), :); closest_diff = data_diff(idx_closest_diff, :);
		
		
		
		
		% ### PLOTTING ###
		if plot
			% Plot distances to H and M using different coloured lines and
			% display distances on screen.
			line_closest_same = plot3([e(1), closest_same(1)], [e(2), closest_same(2)], [e(3), closest_same(3)], 'g-', 'LineWidth', 4);
			pause(timeout);
			line_closest_diff = plot3([e(1), closest_diff(1)], [e(2), closest_diff(2)], [e(3), closest_diff(3)], 'r-', 'LineWidth', 4);
			pause(timeout + 0.05);

			if use_deletions
				delete(sample_p); delete(line_closest_same); delete(line_closest_diff);
			end
		end
		% ### /PLOTTING ###
		
		
		
		% ***************** FEATURE WEIGHTS UPDATE *****************
		% Go over features and update weights.
		for t = 1:size(data, 2)-1
			weights(t) = weights(t) - (abs(e(t) - closest_same(t))/(max_f_vals(t) - min_f_vals(t)))/m + (abs(e(t) - closest_diff(t))/(max_f_vals(t) - min_f_vals(t)))/m;
		end
		% **********************************************************
		
	end
	
	% Rank features.
	[~, p] = sort(weights, 'descend');
	rank = 1:length(weights);
	rank(p) = rank;
	
end