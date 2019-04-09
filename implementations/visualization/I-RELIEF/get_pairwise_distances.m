function [dist_mat] = get_pairwise_distances(data, dist_func)
    % function [dist_mat] = get_pairwise_distances(data, dist_func)
    %
    % Get pairwise distance matrix of exampel in matrix data using distance
    % function dist_func.
    %
    % data --- matrix of training examples
    % dist_func --- distance function to use when comparing examples.
    
    % Compute distance matrix.
    dist_mat = squareform(pdist(data, dist_func));
end