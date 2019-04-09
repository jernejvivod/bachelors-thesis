function [rank, weights] = irelief_animation(data, target, dist_func, max_iter, k_width, conv_condition, initial_w_div)
	% function [weights] = relief_animation(data, m, dist_func, timeout, use_deletions)
	%
	% Create an animation of the basic Relief feature selection algorithm
	% using three dimensional feature space.
	%
	% data --- matrix of training examples
    % target --- classes of training examples
	% dist_func --- weighted distance function for evaluating distance between
	% examples. The function should be able to take two matrices of
	% examples and return a vector of distances between the examples.
	% max_iter --- maximum number of iterations to perform
    % k_width --- kernel width (used in gamma values computation)
    % conv_condition --- threshold for convergence declaration
    % initial_w_div --- value with which to divide the initial weights
    % values.
    %
	% Author: Jernej Vivod


    % Intialize convergence indicator and distance weights for features.
    convergence = false;
    dist_weights = ones(1, size(data, 2))/initial_w_div;

    % Get mean m and mean h vals for all examples.
    mean_m_vals = get_mean_m_vals(data, target);
    mean_h_vals = get_mean_h_vals(data, target);

    % Initialize iteration co[rank, weights] = irelief_animation(data, target, @(x1, x2, w) sum(abs(w.*(x1 - x2).^2).^(1/2), 2), 100, 2.0, 0.0, size(data, 2));unter.
    iter_count = 0;

    % Main iteration loop.
    while iter_count < max_iter && ~convergence

        % Partially apply distance function with weights.
        dist_func_w = @(x1, x2) dist_func(x1, x2, dist_weights);

        % Compute weighted pairwise distance matrix.
        pairwise_dist = get_pairwise_distances(data, dist_func_w);

        % Compute gamma values and compute nu.
        gamma_vals = get_gamma_vals(pairwise_dist, target, @(d) exp(-d/k_width));
        nu = get_nu(gamma_vals, mean_m_vals, mean_h_vals, size(data, 1));

        % Get updated distance weights.
        dist_weights_nxt = max(nu, 0)/norm(max(nu, 0));

        % Check if convergence condition satisfied.
        if sum(abs(dist_weights_nxt - dist_weights) ) < conv_condition
           dist_weights = dist_weights_nxt;
           convergence = true;
        else
           dist_weights = dist_weights_nxt;
           iter_count = iter_count + 1;
        end
    end
    
    % Get feature weights and rank.
    weights = dist_weights;
    [~, p] = sort(dist_weights, 'descend');
    rank = 1:length(dist_weights);
    rank(p) = rank;
end