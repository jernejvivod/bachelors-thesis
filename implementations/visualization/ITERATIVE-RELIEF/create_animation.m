function [] = create_animation(axes, fgr, pauseBtnHandle, resetBtnHandle, timeout)
    
    data = load('rba_test_data2.m');

    % Use deletions
    use_deletions = 1;
    
    min_incl = 2;

    % Create animation and display final feature rank and weights.
    iterative_relief_animation(data(:, 1:end-1), data(:, end), size(data, 1), min_incl,  @(a, b, w) minkowski_dist_weighted(a, b, w, 2), 100, 1, axes, fgr, pauseBtnHandle, resetBtnHandle, timeout, use_deletions);

    % Define weighted minkowski function that takes two vectors or matrices,
    % weights w and the Uspelo mi je implementirati nekaj idej in prebrati članke, ki ste mi jih poslali.parameter p and returns the distance or vector of 
    % distances between the examples.
    function d = minkowski_dist_weighted(a, b, w, p)
        d = sum((abs(w.*(a - b)).^p), 2).^(1/p);
    end
end