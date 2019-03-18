function [r, g, b] = heatmap_col(minimum, maximum, value)
    % function [r, g, b] = heatmap_col(minimum, maximum, value)
    %
    % Return heatmap rgb values for values on interval [minimum, maximum].
    %
    % Args:
    %   minimum ... minimum value
    %   maximum ... maximum value
    %   value   ... value from interval [minimum, maximum]
    %
    % Returns:
    %   
    %
    ratio = 2*(value-minimum)/(maximum - minimum);
    b = max(0, 255*(1 - ratio));
    r = max(0, 255*(ratio - 1));
    g = 255 - b - r;
end