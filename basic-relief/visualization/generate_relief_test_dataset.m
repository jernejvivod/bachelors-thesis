% Construct and visualize a typical dataset where RBA feature selection
% algorithms are especially useful. Save training set in file
% 'rba_test_data.m'.

examples = rand(200, 3);						% Create examples.
target = examples(:, 1) > examples(:, 2);		% Set class value to 1 if first attribute values is greater than second attribute value.
yes_examples = examples(logical(target), :);	% Split examples based on class.
no_examples = examples(not(logical(target)), :);
dataset = [examples, target];
save('rba_test_data.m', 'dataset', '-ascii');

% Plot the dataset in 3 dimensions.
figure(1); hold on; axis equal; grid on; xlabel('a'); ylabel('b'); zlabel('c');
title('Sample Dataset for RBD Feature Selection')
scatter3(yes_examples(:, 1), yes_examples(:, 2), yes_examples(:, 3), 'r', 'filled');
scatter3(no_examples(:, 1), no_examples(:, 2), no_examples(:, 3), 'b', 'filled');
view(30, 20)

% Plot separations of classes by each feature.
figure(2);
subplot(3, 1, 1); hold on; title('Class Separation by Feature a')
feature_proj1([examples, target], 1, 'a');

subplot(3, 1, 2); hold on; title('Class Separation by Feature b')
feature_proj1([examples, target], 2, 'b');

subplot(3, 1, 3); hold on; title('Class Separation by Feature c')
feature_proj1([examples, target], 3, 'c');

% Plot separation of classes by each pair of features.
figure(3);
subplot(1, 3, 1); hold on; title('Class Separation by Features a and b')
feature_proj2([examples, target], [1, 2], ['a', 'b']);

subplot(1, 3, 2); hold on; title('Class Separation by Features a and c')
feature_proj2([examples, target], [1, 3], ['a', 'c']);

subplot(1, 3, 3); hold on; title('Class Separation by Features b and c')
feature_proj2([examples, target], [2, 3], ['b', 'c']);

% TODO document functions
function [] = feature_proj1(data, idx_feature, feature_name)
	% Plot class separation with respect to specified single feature on the
	% number line.
	%
	% data -- matrix containing the training examples and their class as
	% the last column
	%
	% idx_feature -- index of the column containing the feature to be
	% plotted
	%
	% feature_name -- name of the feature being plotted (used to label the axis)

	target = data(:, end);  % Get target/class value vector.
	feature_values1 = data(logical(target), idx_feature);  % separate feature values based on target/class.
	feature_values2 = data(not(logical(target)), idx_feature);
	plot(feature_values1, 0, 'r*', 'MarkerSize', 7);  % Plot data.
	plot(feature_values2, 0, 'b.', 'MarkerSize', 10);
	xlabel(feature_name);
end

function [] = feature_proj2(data, idx_features, feature_names)
	% Plot class separation with respect to specified pair of features.
	%
	% data -- matrix containing the training examples and their class as
	% the last column
	%
	% idx_features -- 1x2 matrix of column indices containing the features
	% to be plotted.
	%
	% feature_names -- names of the features being plotted (used to label the axes)

	target = data(:, end);  % Get target/class value vector.
	feature_values1 = data(logical(target), idx_features);  % separate feature values based on target/class
	feature_values2 = data(not(logical(target)), idx_features);
	% Plot data.
	scatter(feature_values1(:, 1), feature_values1(:, 2), 'r', 'filled');
	scatter(feature_values2(:, 1), feature_values2(:, 2), 'b', 'filled');
	xlabel(feature_names(1)); ylabel(feature_names(2));
end