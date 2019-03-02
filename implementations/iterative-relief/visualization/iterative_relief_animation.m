function [rank, dist_weights] = iterative_relief_animation(data, target, m, min_incl, dist_func, max_iter, plot, timeout, use_deletions)
	% function [weights] = iterative_relief_animation(data, target, m, min_incl, dist_func, max_iter, plot, timeout, use_deletions)
	%
	% Create an animation of the Iterative Relief feature selection algorithm
	% using three dimensional feature space.
	%
	% data --- matrix of training data features in rows
	% target --- matrix containing the class values for each example
	% m --- the m parameter (example sample size)
	% min_incl --- the minimum number of examples from same and other
	% classes that a hypersphere centered at each examples should contain.
	% dist_func --- distance function for evaluating distance between
	% examples. The function should be able to take two matrices of
	% examples and return a vector of distances between the examples. The
	% distance function should accept a weights parameter.
	% max_iter --- Maximum number of iterations to compute.
	% plot --- if set to 0, the function only computes the weights and
	% ranks without plotting.
	% timeout --- duration of pause between graphical object plotting function calls.
	% use_deletions --- logical value that specifies whether to delete the
	% computation visualization when moving onto next example in the
	% sample.
	%
	% Author: Jernej Vivod
	
	
	% Set parameter values if not present in call.
	if nargin < 9
		use_deletions = 1;
		if nargin < 8
			timeout = 0.3;
			if nargin < 7
				plot = 0;
			end
		end
	end
	
	
	
	% ### PLOTTING ###
	if plot
		% Create scatter plot of data
		figure(1); hold on;
		scatter3(data(:, 1), data(:, 2), data(:, 3), 30, categorical(target), 'filled');
		xlabel('a'); ylabel('b'); zlabel('c'); view(30, 50); grid on;
		axis equal; xlim([0, 1]); ylim([0, 1]); hold all;
		pause on;
	end
	% ### /PLOTTING ###
	
	
	
	% Get minimum radius of hypersphere so that each hypersphere with
	% center at one of the examples will contain at least min_incl examples
	% from same class and min_incl examples from another class.
	min_r = min_radius(min_incl, data, target);

	
	% Initialize all distance weights to 0.
	dist_weights = ones(1, size(data, 2));  % Initialize weights.
	
	% Compute minimum and maximum feature values.
	% max_f_vals = max(data(:, 1:end-1));
	% min_f_vals = min(data(:, 1:end-1));
	
	% Initialize iteration counter and convergence indicator to 0.
	iter_count = 0;
	convergence = 0;
	feature_weights_prev = zeros(1, size(data, 2));
	
	% --- Main iteration loop ---
	while iter_count < max_iter && ~convergence
		
		% Increment iteration counter.
		iter_count = iter_count + 1;
		
		% Reset all feature weights to 0.
		feature_weights = zeros(1, size(data, 2));
		
		% Sample m examples from training set (get indices).
		idx_sampled = randsample(1:size(data, 1), m);
		
		% Go over examples in sample.
		for idx = 1:10



			% ### PLOTTING ###
			if plot
				% Display current weight values.
				hT = title({'Iterative Relief Algorithm Visualization', sprintf('$$ weights = [%.3f, %.3f, %.3f] $$', dist_weights(1), dist_weights(2), dist_weights(3))},'interpreter','latex');
				set(hT, 'FontSize', 17);
			end
			% ### /PLOTTING ###



			% Get next sampled example.
			e = data(idx, :);



			% ### PLOTTING ###
			if plot
				% Mark selected example.
				sample_p = plot3(e(1), e(2), e(3), 'ro', 'MarkerSize', 10);
				pause(timeout);
				
				% Plot hypersphere.
				[x, y, z] = sphere(128);
				axis manual;
				hypersph = surf(min_r*x + e(1), min_r*y + e(2), min_r*z + e(3));
				set(hypersph, 'FaceAlpha', 0.08);
				shading interp;
				light1 = light('Position',[-1 2 0],'Style','local');
				set(hypersph, 'FaceColor', [0 0.1 0.8]);
				
				pause(timeout);
			end
			% ### /PLOTTING ###
			
			
			
			% Get examples from same class in hypersphere with center at e
			same_in_hypsph = sum((data(target == target(idx), :) - e).^2, 2) <= min_r^2;
			data_same_aux = data(target == target(idx), :); data_same = data_same_aux(same_in_hypsph, :);
			
			% Compute weighted distances to examples in hypersphere from same class.
			dist_same = dist_func(e, data_same, dist_weights);
			
			% Get examples from other classes in hypersphere with center at e
			other_in_hypsph = sum((data(target ~= target(idx), :) - e).^2, 2) <= min_r^2;
			data_other_aux = data(target ~= target(idx), :); data_other = data_other_aux(other_in_hypsph, :);
			
			% Compute weighted distances to examples in hypersphere from different class.
			dist_other = dist_func(e, data_other, dist_weights);
			
			
			
			% ### PLOTTING ###
			if plot
				% Plot distances to examples from same class and examples from other classes inside hypersphere.
				line_ctr = 1;
				lines_same = cell(1, length(same_in_hypsph));
				for same_nxt = data_same'
					lines_same{line_ctr} = plot3([e(1), same_nxt(1)], [e(2), same_nxt(2)], [e(3), same_nxt(3)], 'g-', 'LineWidth', 2);
					pause(timeout);
					line_ctr = line_ctr + 1;
				end
				pause(timeout);

				line_ctr = 1;
				lines_other = cell(1, length(other_in_hypsph));
				for other_nxt = data_other'
					lines_other{line_ctr} = plot3([e(1), other_nxt(1)], [e(2), other_nxt(2)], [e(3), other_nxt(3)], 'r-', 'LineWidth', 2);
					pause(timeout);
					line_ctr = line_ctr + 1;
				end

				if use_deletions
					delete(sample_p); delete(hypersph); cellfun(@delete, lines_same); cellfun(@delete, lines_other);
					delete(light1);
				end
			end
			% ### /PLOTTING ###
			
			

			% ***************** FEATURE WEIGHTS UPDATE *****************
			w_miss = max(0, 1 - (dist_other.^2/min_r^2));
			w_hit = max(0, 1 - (dist_same.^2/min_r^2));

			numerator1 = sum(abs(e - data_other) .* w_miss, 1);
			denominator1 = sum(w_miss) + eps;

			numerator2 = sum(abs(e - data_same) .* w_hit, 1);
			denominator2 = sum(w_hit) - 1 + eps;  % Subtract weight of sampled example to itself.

			feature_weights = feature_weights + numerator1/denominator1 - numerator2/denominator2;
			% **********************************************************
			
		end
		
		% Update distance weights by feature weights.
		dist_weights = dist_weights + feature_weights;
		if sum(abs(feature_weights - feature_weights_prev)) < 0.01
			convergence = 1;
		end
		feature_weights_prev = feature_weights;
		
	end
	
	% Rank features.
	[~, p] = sort(dist_weights, 'descend');
	rank = 1:length(dist_weights);
	rank(p) = rank;
	
end