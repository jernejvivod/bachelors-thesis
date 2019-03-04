dataset = load('rba_test_data3.mat');
target = dataset.dataset.target; data = dataset.dataset.data;

lim_noise = 10;
noisy_data =  [data, rand(size(data, 1), lim_noise)];

res = zeros(lim_noise, 4);

figure; hold on; title(sprintf('Iterative Relief Weights for Monotonic and Non-monotonic\nRelevant Feature with Respect to Number of Noise Features'));
xlim([0, lim_noise]); ylim([-50, 300]); xlabel('number of noise features'); ylabel('Weight');
pause(0.5);
for k = 0:lim_noise
	[rank, weights] = iterative_relief_animation(noisy_data(:, 1:2+k), target, size(data, 1), 2, @(a, b, w) minkowski_dist_weighted(a, b, w, 2), 100, 0);
	res(k+1, 1) = weights(1);
	res(k+1, 2) = weights(2);
	if k > 0
		res(k+1, 3) = mean(weights(3:end));
		res(k+1, 4) = max(weights(3:end));
	else
		res(k+1, 3) = NaN;
		res(k+1, 4) = NaN;
	end
	plot(k, res(k+1, 1), 'b.'); plot(k, res(k+1, 2), 'g.');
	plot(k, res(k+1, 3), 'r.'); plot(k, res(k+1, 4), 'y.');
	t1 = text(k, res(k+1, 1) + 10, 'monotonic relevant feature weight');
	t2 = text(k, res(k+1, 2) + 10, 'non-monotonic relevant feature weight');
	t3 = text(k, res(k+1, 3) + 10, 'average noise feature weight');
	t4 = text(k, res(k+1, 4) + 20, 'maximum noise feature weight');
	pause(0.1);
	delete(t1); delete(t2); delete(t3); delete(t4);
end

figure; title(sprintf('Iterative Relief Weights for Monotonic and Non-monotonic Feature\nwith Respect to Number of Noise Features'));	
hold on; xlabel('Number of Noise Features'); ylabel('Weight');
plot(0:lim_noise, res(:, 1), 'b-');
plot(0:lim_noise, res(:, 2), 'g-');
plot(0:lim_noise, res(:, 3), 'r-');
plot(0:lim_noise, res(:, 4), 'y-');

legend('monotonic relevant feature weight',...
	'non-monotonic relevant feature weight',...
	'average noise features weight',...
	'maximal noise feature weight',...
	'Location', 'southeast');

xlim([0, lim_noise]); xlabel('Number of Noise Features'); ylabel('Weight');


% Define weighted minkowski function that takes two vectors or matrices,
% weights w and the parameter p and returns the distance or vector of 
% distances between the examples.
function d = minkowski_dist_weighted(a, b, w, p)
	d = sum((abs(w.*(a - b)).^p), 2).^(1/p);
end