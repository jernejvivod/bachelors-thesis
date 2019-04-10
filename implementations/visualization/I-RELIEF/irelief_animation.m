function [rank, weights] = irelief_animation(data, target, dist_func, max_iter, k_width, conv_condition, initial_w_div, plot, axes, fgr, pauseBtnHandle, resetBtnHandle, timeout)
	% function [weights] = relief_animation(data, m, dist_func, timeout, use_deletions)
	%
	% Create an animation of the basic Relief feature selection algorithm
	% using three dimensional feature space.
	%
	% data --- matrix of training examples
    % target --- classes of training examples
	% dist_func --- weighted distance function for evaluating distance between
	% examples. The function should be able to take two matrices of
	% examples and return a vector of distances between the examples.
	% max_iter --- maximum number of iterations to perform
    % k_width --- kernel width (used in gamma values computation)
    % conv_condition --- threshold for convergence declaration
    % initial_w_div --- value with which to divide the initial weights
    % values.
    %
	% Author: Jernej Vivod

    try  % try/catch statement that will catch errors occuring when closing app.
        
        
        % ### PLOTTING ###
        if plot
            % Set axis labels.
            axes.XLabel.String = 'a'; axes.YLabel.String = 'b'; axes.ZLabel.String = "c";
            % axes.XLim = [0, 1]; axes.YLim = [0, 1]; hold(axes, 'all');
            % axis(axes, 'equal');
            % Set pause.
            pause on;
        end
        % ### /PLOTTING ###

        % Intialize convergence indicator and distance weights for features.
        convergence = false;
        dist_weights = ones(1, size(data, 2))/initial_w_div;

        % Get mean m and mean h vals for all examples.
        mean_m_vals = get_mean_m_vals(data, target);
        mean_h_vals = get_mean_h_vals(data, target);

        % Initialize iteration co[rank, weights] = irelief_animation(data, target, @(x1, x2, w) sum(abs(w.*(x1 - x2).^2).^(1/2), 2), 100, 2.0, 0.0, size(data, 2));unter.
        iter_count = 0;

        % Main iteration loop.
        while iter_count < max_iter && ~convergence
            
            
            % ### PLOTTING ###
            if plot
                % Display current weight values.
                hT = title(axes, {'I-Relief Algorithm Visualization', sprintf('$$ weights = [%.3f, %.3f, %.3f] $$', dist_weights(1), dist_weights(2), dist_weights(3))},'interpreter','latex');
                set(hT, 'FontSize', 17);
            end
            % ### /PLOTTING ###
            
            

            % Partially apply distance function with weights.
            dist_func_w = @(x1, x2) dist_func(x1, x2, dist_weights);

            % Compute weighted pairwise distance matrix.
            pairwise_dist = get_pairwise_distances(data, dist_func_w);




            %%% PLOTTING %%%
            if plot
                pause on;

                if timeout > 0.0
                    pause(timeout);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                    if get(pauseBtnHandle, 'Value')
                        while get(pauseBtnHandle, 'Value')
                           pause(0.5);
                           if get(resetBtnHandle, 'Flag')
                              return; 
                           end
                        end
                    end

                else
                    uiwait(fgr);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                end

                vis_sub = 180;  % Number of examples not to plot (for speed).
                lines = gobjects(1, size(pairwise_dist, 2));
                for idx1 = 1:size(pairwise_dist, 1)-vis_sub
                    for idx2 = 1:size(pairwise_dist, 1)
                        if idx1 ~= idx2
                            [r, g, b] = heatmap_cols(min(pairwise_dist(:)), max(pairwise_dist(:)), pairwise_dist(idx1, idx2));
                            lines(idx2) = line(axes, [data(idx1, 1), data(idx2, 1)], [data(idx1, 2), data(idx2, 2)], [data(idx1, 3), data(idx2, 3)], 'Color', [r, g, b]/255);
                        end
                    end
                    pause(0.01)
                    for i = 1:length(lines)
                        delete(lines(i))
                    end
                    if get(pauseBtnHandle, 'Value')
                        while get(pauseBtnHandle, 'Value')
                           pause(0.5);
                           if get(resetBtnHandle, 'Flag')
                              return; 
                           end
                        end
                    end
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                end

                if timeout > 0.0
                    pause(timeout);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                    if get(pauseBtnHandle, 'Value')
                        while get(pauseBtnHandle, 'Value')
                           pause(0.5);
                           if get(resetBtnHandle, 'Flag')
                              return; 
                           end
                        end
                    end

                else
                    uiwait(fgr);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                end
            end
            %%% /PLOTTING %%% 

            
            % ### PLOTTING ###
            if plot
                % Display current weight values.
                hT = title(axes, {'Computing Gamma Values...'}, 'interpreter', 'latex');
                set(hT, 'FontSize', 17);
            end
            % ### /PLOTTING ###

            % Compute gamma values and compute nu.
            gamma_vals = get_gamma_vals(pairwise_dist, target, @(d) exp(-d/k_width));
            
            


            %%% PLOTTING %%%
            if plot
                % plot each gamma value as a sphere centerd on a point.
                spheres = gobjects(1, size(data, 1));
                texts = gobjects(1, size(data, 1));
                for idx = 1:size(data, 1)
                    % Plot hypersphere.
                    [x, y, z] = sphere(10);
                    axis(axes, 'manual');
                    spheres(idx) = surf(axes, 0.05*x + data(idx, 1), 0.05*y + data(idx, 2), 0.05*z + data(idx, 3));
                    set(spheres(idx), 'FaceAlpha', 0.1);
                    shading(axes, 'interp');
                    texts(idx) = text(axes, data(idx, 1), data(idx, 2), data(idx, 3), sprintf("%.3f", gamma_vals(idx)), 'FontSize', 6);
                end
                for idx = 1:size(data, 1)
                   [r, g, b] = heatmap_cols(min(gamma_vals(:)), max(gamma_vals(:)), gamma_vals(idx));
                   set(spheres(idx), 'FaceColor', [r, g, b]/255); 
                end
                
                % Display current weight values.
                hT = title(axes, {'Gamma Values'}, 'interpreter', 'latex');
                set(hT, 'FontSize', 17);
                
                pause(1.5)

                if timeout > 0.0
                    pause(timeout);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                    if get(pauseBtnHandle, 'Value')
                        while get(pauseBtnHandle, 'Value')
                           pause(0.5);
                           if get(resetBtnHandle, 'Flag')
                              return; 
                           end
                        end
                    end

                else
                    uiwait(fgr);
                    if get(resetBtnHandle, 'Flag')
                        return; 
                    end
                end

                for idx = 1:length(spheres)
                    delete(spheres(idx))
                    delete(texts(idx))
                end
            end

            %%% /PLOTTING %%%


            % Compute nu value.
            nu = get_nu(gamma_vals, mean_m_vals, mean_h_vals, size(data, 1));



            % Get updated distance weights.
            dist_weights_nxt = max(nu, 0)/norm(max(nu, 0));

            % Check if convergence condition satisfied.
            if sum(abs(dist_weights_nxt - dist_weights) ) < conv_condition
               dist_weights = dist_weights_nxt;
               convergence = true;
            else
               dist_weights = dist_weights_nxt;
               iter_count = iter_count + 1;
            end
        end

        % Get feature weights and rank.
        weights = dist_weights;
        [~, p] = sort(dist_weights, 'descend');
        rank = 1:length(dist_weights);
        rank(p) = rank;
    catch
        return
    end
end

function [r, g, b] = heatmap_cols(minimum, maximum, value)
    ratio = 2*(value-minimum)/(maximum-minimum);
    b = ceil(max(0, 255*(1 - ratio)));
    r = ceil(max(0, 255*(ratio - 1)));
    g = 255 - b - r;
end