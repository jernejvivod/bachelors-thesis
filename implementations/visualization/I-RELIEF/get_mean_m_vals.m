function [mean_m] = get_mean_m_vals(data, classes)
    % function [mean_m] = get_mean_m_vals(data, classes)
    %
    % get mean m values for each example in dataset. Mean m value of an example is the average
    % difference of examples with a different class and this example.
    %
    % data --- matrix of training examples.
    % classes --- classes of training examples.
    
    % Allocate matrix for storing the mean m values.
    mean_m = zeros(size(data));
    
    % Compute matrix of pairwise distances.
    pairwise_diff = abs(reshape(data', [1, size(data, 2), size(data, 1)]) - data);
    
    % Go over submatrices of pairwise distances for each example.
    for idx = 1:size(pairwise_diff, 3)
        r = pairwise_diff(:, :, idx);
        % Compute mean m value for each example.
        mean_m(idx, :) = mean(r(classes ~= classes(idx), :), 1);
    end
end