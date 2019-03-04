% Create the random points.
rand_points = rand(1000, 100);
% Allocate matrix for storing results.
res = zeros(size(rand_points, 2), 2);

% Go over dimensionalities.
for k = 1:size(rand_points, 2)
	pts_nxt = rand_points(:, 1:k);
	pdist_mat1 = squareform(pdist(pts_nxt, 'euclidean'));
	pdist_mat2 = squareform(pdist(pts_nxt, 'cityblock'));
	
	% Remove diagonal elements (distances from point to itself).
	pdist_mat1_noself = reshape(nonzeros(pdist_mat1'), size(pdist_mat1, 2)-1, [])';
	pdist_mat2_noself = reshape(nonzeros(pdist_mat2'), size(pdist_mat2, 2)-1, [])';
	
	% Compute results.
	avg_ratio1 = mean(min(pdist_mat1_noself, [], 2)./max(pdist_mat1_noself, [], 2));
	avg_ratio2 = mean(min(pdist_mat2_noself, [], 2)./max(pdist_mat2_noself, [], 2));
	res(k, :) = [avg_ratio1, avg_ratio2];
end

% Plot results.
figure; hold on;
plot(1:size(rand_points, 2), res(:, 1)); plot(1:size(rand_points, 2), res(:, 2));
legend('Evklidska razdalja', 'Manhattanska Razdalja');
title(sprintf('Povprečno razmerje med razdaljo do najbližje točke in razdaljo do\nnajbolj oddaljene točke v odvisnosti od dimenzionalnosti prostora'));
xlabel('dimenzionalnost hiperkocke');
ylabel(sprintf('povprečne razdalja med %d naključnimi točkami', size(rand_points, 1)));