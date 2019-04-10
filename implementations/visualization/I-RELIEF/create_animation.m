function [] = create_animation(axes, fgr, pauseBtnHandle, resetBtnHandle, timeout)
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
    data_raw = load('rba_test_data2.m');
    data = data_raw(:, 1:end-1);
    target = data_raw(:, end);
    
    % Perform plotting.
    plot = true;

    % Create animation and display final feature weights.
    irelief_animation(data, target, @(x1, x2, w) sum(abs(w.*(x1-x2)).^2, 2).^(1/2), 100, 2.0, 0.0, size(data, 1), plot, axes, fgr, pauseBtnHandle, resetBtnHandle, timeout);
end