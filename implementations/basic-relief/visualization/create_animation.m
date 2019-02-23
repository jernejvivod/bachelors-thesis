function [] = create_animation()
	% function [] = create_animation()
	%
	% Display animation for the basic Relief algorithm.
	% Called with no arguments. Animation properties are set in the
	% function body.
	
	
	% Define minkowski function that takes two vectors or matrices
	% and the parameter p and returns the distance or vector of distances
	% between the examples.
	function d = minkowski_dist(a, b, p)
		d = sum(abs(a - b).^p, 2).^(1/p);
	end

	% Test data 1
	data1 = [0 0 0 0; 0 0 1 0; 0 1 1 1; 0 1 0 1; 1 0 0 1; 1 0 1 1; 1 1 0 0; 1 1 0 0];
	
	% Test data 2
	data2 = load('rba_test_data.m');
	
	% Use deletions
	use_deletions = 1;
	
	% Set example animation timeout
	timeout = 0.3;
	
	% Create animation and display final feature weights.
	weights = relief_animation(data2, 10,  @(a, b) minkowski_dist(a, b, 2), timeout, use_deletions);
	fprintf('weights = [%.3f, %.3f, %.3f]\n', weights);
end