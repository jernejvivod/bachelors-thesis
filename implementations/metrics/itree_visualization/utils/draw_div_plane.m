function [] = draw_div_plane(dim, val, space_dims, axes)
    % function [] = draw_div_plane(dim, val, space_dims)
    %
    % Draw a divising plane splitting the space by dimension dim
    % at value val.
    %
    % Args:
    %   dim        ... dimension to split on
    %   val        ... splitting value
    %   space_dims ... dimensions of the space being split. A 3x2 matrix
    %                  where the ith row represents the ith dimension. The
    %                  first column represents the lower bound and the second 
    %                  column the upper bound.
    %   axes       ... axes on which to plot
    %
    % Returns:
    %   unit/void

    
    % Switch on split dimension and plot.
    switch dim
        case 1
            [Y, Z] = meshgrid(linspace(space_dims(2, 1), space_dims(2, 2), 5), linspace(space_dims(3, 1), space_dims(3, 2), 5));
            x = repmat(val, size(Y));
            s = surf(axes, x, Y, Z, 'FaceColor', 'b');
            alpha(s, 0.1); s.EdgeColor = 'none';
        case 2
            [X, Z] = meshgrid(linspace(space_dims(1, 1), space_dims(1, 2), 5), linspace(space_dims(3, 1),space_dims(3, 2), 5));
            y = repmat(val, size(X));
            s = surf(axes, X, y, Z, 'FaceColor', 'b');
            alpha(s, 0.1); s.EdgeColor = 'none';
        case 3
            [X, Y] = meshgrid(linspace(space_dims(1, 1), space_dims(1, 2), 5), linspace(space_dims(2, 1), space_dims(2, 2), 5));
            z = repmat(val, size(X));
            s = surf(axes, X, Y, z, 'FaceColor', 'b');
            alpha(s, 0.1); s.EdgeColor = 'none';
        otherwise
    end
end