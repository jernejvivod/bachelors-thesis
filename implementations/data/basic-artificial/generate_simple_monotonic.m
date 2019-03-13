data = feval(@(x) x, linspace(0, 1, 100));
target = double(data > 0.5);

% Construct and save dataset.
dataset.data = data';
dataset.target = target';
save('simple_monotonic.mat', 'dataset');