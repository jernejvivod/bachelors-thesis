function [mean_h] = get_mean_h_vals(data, classes)
    % function [mean_m] = get_mean_m_vals(data, classes)
    % 
    % get mean m values for each example in dataset. Mean m value of an example is the average
    % difference of examples with a different class and this example.
    %
    %
    % data --- matrix of training examples.
    % classes --- classes of training examples.

  
    % Allocate matrix for storing the mean h values.
    mean_h = zeros(size(data));
    
    % Compute matrix of pairwise distances.
    pairwise_diff = abs(reshape(data', [1, size(data, 2), size(data, 1)]) - data);
    
    % Go over submatrices of pairwise distances for exah example.
    for idx = 1:size(pairwise_diff, 3)
        r = pairwise_diff(:, :, idx);
        % Get mask of classes that equal the class of the current example.
        msk = classes == classes(idx);
        % Exclude current example from mask.
        msk(idx) = false;
        mean_h(idx, :) = mean(r(msk, :), 1);
    end
end