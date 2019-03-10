% Construct and visualize a typical dataset where RBA feature selection
% algorithms are especially useful. Save training set in file
% 'rba_test_data2.m'.

examples = rand(200, 3);						% Create examples.

c1 = examples(:, 1) > examples(:, 2);
c2 = examples(:, 1) + examples(:, 2) < 1;
c3 = c1 + c2;
c4 = c3 + c2;
target = c4;

dataset = [examples, target];
save('rba_test_data2.m', 'dataset', '-ascii');

% Plot the dataset in 3 dimensions.
figure(1); hold on; axis equal; grid on; xlabel('a'); ylabel('b'); zlabel('c');
title('Sample Dataset for RBD Feature Selection')
scatter3(dataset(:, 1), dataset(:, 2), dataset(:, 3), 30, categorical(dataset(:, end)), 'filled');
view(30, 20); axis equal;

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
subplot(2, 2, 1); hold on; title('Class Separation by Features a and b')
feature_proj2([examples, target], [1, 2], ['a', 'b']);

subplot(2, 2, 2); hold on; title('Class Separation by Features a and c')
feature_proj2([examples, target], [1, 3], ['a', 'c']);

subplot(2, 2, 3); hold on; title('Class Separation by Features b and c')
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
	
	% Plot data.
	scatter(data(:, idx_feature), zeros(size(data, 1), 1), 30, categorical(data(:, end)), 'filled');
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
	
	% Plot data.
	scatter(data(:, idx_features(1)), data(:, idx_features(2)), 30, categorical(data(:, end)), 'filled');
	xlabel(feature_names(1)); ylabel(feature_names(2));
end