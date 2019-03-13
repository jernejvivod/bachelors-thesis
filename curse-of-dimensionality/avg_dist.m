% Create the random points.
rand_points = rand(100, 100);
% Allocate matrix for storing results.
res = zeros(size(rand_points, 1), 2);

% Go over dimensionalities.
for k = 1:size(rand_points, 1)
	pts_nxt = rand_points(:, 1:k);
	pdist_mat1 = squareform(pdist(pts_nxt, 'euclidean'));
	pdist_mat2 = squareform(pdist(pts_nxt, 'cityblock'));
	avg_dist_val1 = sum(sum(pdist_mat1, 2)./(size(pdist_mat1, 2)-1));
	avg_dist_val2 = sum(sum(pdist_mat2, 2)./(size(pdist_mat2, 2)-1));
	res(k, :) = [avg_dist_val1, avg_dist_val2];
end

% Plot results.
figure('Renderer', 'painters', 'Position', [10 10 800 450]); hold on;
plot(1:size(rand_points, 2), res(:, 1)); plot(1:size(rand_points, 2), res(:, 2));
legend('Evklidska razdalja', 'Manhattanska Razdalja');
title(sprintf('Povprečna razdalja med naključnimi točkami v hiperkocki\nz stranico dolžine 1 v odvisnosti od dimenzionalnosti hiperkocke'));
xlabel('dimenzionalnost hiperkocke');
ylabel(sprintf('povprečne razdalja med %d naključnimi točkami', size(rand_points, 1)));