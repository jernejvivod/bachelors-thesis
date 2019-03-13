data = feval(@(x) normpdf(x, 0.5, 0.3), linspace(0, 1, 100));
target = double(data > 0.8);

% Construct and save dataset.
dataset.data = data';
dataset.target = target';
save('simple_nonmonotonic.mat', 'dataset');