% Test data 1
data1 = [0 0 0 0; 0 0 1 0; 0 1 1 1; 0 1 0 1; 1 0 0 1; 1 0 1 1; 1 1 0 0; 1 1 0 0];

% Set m and k parameters.
m = 50; min_incl = 1;

% Test data 2
data2 = load('rba_test_data.m');

% Test data 3
data3 = load('rba_test_data2.m');

% Use deletions
use_deletions = 1;

% Set example animation timeout
timeout = 0.01;

% Create animation and display final feature rank and weights.
[rank, weights] = iterative_relief_animation(data3(:, 1:end-1), data3(:, end), size(data3, 1), min_incl,  @(a, b, w) minkowski_dist_weighted(a, b, w, 2), 100, 1, timeout, use_deletions);

fprintf('rank = [%d, %d, %d]\n', rank);
fprintf('weights = [%.3f, %.3f, %.3f]\n', weights);

% Define weighted minkowski function that takes two vectors or matrices,
% weights w and the Uspelo mi je implementirati nekaj idej in prebrati članke, ki ste mi jih poslali.parameter p and returns the distance or vector of 
% distances between the examples.
function d = minkowski_dist_weighted(a, b, w, p)
	d = sum((abs(w.*(a - b)).^p), 2).^(1/p);
end