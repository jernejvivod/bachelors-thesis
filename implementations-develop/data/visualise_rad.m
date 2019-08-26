load data
load target

figure; hold on;
scatter(data(target == 1, 1), data(target == 1, 2), 'b', 'filled');
scatter(data(target == 0, 1), data(target == 0, 2), 'r', 'filled');
xlabel('\nu_{1}', 'interpreter', 'tex')
ylabel('\nu_{2}', 'interpreter', 'tex')
sel_example = data(55, :)
plot(sel_example(1), sel_example(2), 'ko', 'MarkerSize', 11)

choice = "SURF";

if choice == "SURF"
		mu = mean(pdist(data));
		viscircles(sel_example, mu, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1);

elseif choice == "MultiSURF*"

		% Distances to selected example
		dist_mat = squareform(pdist(data));
		dists_other = dist_mat(55, 1:end ~= 55);
		mu = mean(dists_other);
		sig = std(dists_other)
		viscircles(sel_example, mu, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1);
		viscircles(sel_example, mu-sig, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1);
		viscircles(sel_example, mu+sig, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1);


elseif choice == "MultiSURF"
		% Distances to selected example
		dist_mat = squareform(pdist(data));
		dists_other = dist_mat(55, 1:end ~= 55);
		mu = mean(dists_other);
		sig = std(dists_other)
		viscircles(sel_example, mu, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 1);
		viscircles(sel_example, mu-sig, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 1);

end

axis equal;
xlim([-0.05, 1.05]);
ylim([-0.05, 1.05]);
