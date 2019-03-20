function [] = create_animation(axes, fgr, stopBtnHandle, pauseBtnHandle, timeout)
    % function [] = create_animation(axes, fgr, stopBtnHandle, resetBtnHandle, timeout)
	%
	% Create an animation of the basic Relief feature selection algorithm
	% using three dimensional feature space. Mean to be called by the
	% front-end GUI.
	%
	% axes           ... the UIAxes object from the GUI representing the axes in the GUI
	% fgr            ... the UIFigure object from the GUI representing the figure in
	% the GUI
    % pauseBtnHandle ... handle of the pause button in the GUI front-end
    % resetBtnHandle ... handle of the reset button in the GUI front-end
    % timeout        ... pause duration between plotrings
    %
	% Author: Jernej Vivod

    % load data
    data = load('rba_test_data2.m');

    % Use deletions
    use_deletions = 1;

    % Create animation and display final feature weights.
    relief_animation(data, size(data, 1),  @(a, b) minkowski_dist(a, b, 2), 1, axes, fgr, stopBtnHandle, pauseBtnHandle, timeout, use_deletions);

    % Define minkowski function that takes two vectors or matrices
    % and the parameter p and returns the distance or vector of distances
    % between the examples.
    function d = minkowski_dist(a, b, p)
        d = sum(abs(a - b).^p, 2).^(1/p);
    end
end