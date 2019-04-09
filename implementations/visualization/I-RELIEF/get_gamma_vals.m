function [gamma_vals] = get_gamma_vals(dist_mat, classes, kern_func)
    % function [gamma_vals] = get_gamma_vals(dist_mat, classes, kern_func)
    %
    % For each example get probability of it being an otlier.
    % Function depends on weights (through distance matrix).
    %
    %
    % dist_mat --- pairwise distance matrix between training examples
    % classes --- classes of training examples.
    % kern_func --- kernel function (see article)
    
    
    % Compute probabilities of examples being outliers.
    po_vals = zeros(size(dist_mat, 1), 1);
    
    % Go over rows of the pariwise distance matrix.
    for idx = 1:size(dist_mat, 1)
        r = dist_mat(idx, :);
        numerator = sum(kern_func(r(classes ~= classes(idx))));
        msk = 1:size(dist_mat, 1) ~= idx;
        denominator = sum(kern_func(r(msk)));
        po_vals(idx) = numerator/denominator;
    end
    
    % Get gamma values.
    gamma_vals = 1 - po_vals;
end