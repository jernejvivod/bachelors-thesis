% Interval from domain used in constructing the feature values
base = linspace(0, 5, 1000);

% Construct monotonic feature.
f1 = feval(@(x) 0.1*x + 0, base);
% Construct non-monotonic feature.
f2 = normpdf(base, 3, 0.8);
% Construct data matrix.
data = [f1', f2'];

% Construct target variable values vector.
target = double(f2 > f1);

% Construct and save dataset.
dataset.data = data;
dataset.target = target';
save('rba_test_data3.mat', 'dataset');