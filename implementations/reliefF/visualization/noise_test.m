dataset = load('rba_test_data3.mat');
target = dataset.dataset.target; data = dataset.dataset.data;

lim_noise = 100;
noisy_data =  [data, rand(size(data, 1), lim_noise)];

res = zeros(lim_noise, 3);

figure; hold on; title(sprintf('ReliefF Weights for Monotonic and Non-monotonic Feature\nwith Respect to Number of Noise Features'));	
xlim([0, lim_noise]); ylim([0, 1]); xlabel('Number of Noise Features'); ylabel('Weight');
pause(0.5);
for k = 0:lim_noise
	[~, weights] = relieff_animation([noisy_data(:, 1:2+k), target], size(data, 1), 3, @(a, b) minkowski_dist(a, b, 2), 0);
	res(k+1, 1) = weights(1);
	res(k+1, 2) = weights(2);
	res(k+1, 3) = mean(weights(3:end));
	plot(k, res(k+1, 1), 'b.'); plot(k, res(k+1, 2), 'g.');
	plot(k, mean(weights(3:end)), 'r.');
	t1 = text(k, res(k+1, 1) + 0.05, 'monotonic feature weight');
	t2 = text(k, res(k+1, 2) + 0.05, 'non-monotonic feature weight');
	t3 = text(k, res(k+1, 3) + 0.05, 'average noise feature weight');
	pause(0.1);
	delete(t1); delete(t2); delete(t3);
end

figure; title(sprintf('ReliefF Weights for Monotonic and Non-monotonic Feature\nwith Respect to Number of Noise Features'));
hold on; xlabel('Number of Noise Features'); ylabel('Weight');
plot(1:lim_noise+1, res(:, 1), 'b-');
plot(1:lim_noise+1, res(:, 2), 'g-');
plot(1:lim_noise+1, res(:, 3), 'r-');
legend('monotonic feature weight', 'non-monotonic feature weight', 'average noise features weight', 'Location', 'southeast');
xlim([0, lim_noise]); xlabel('Number of Noise Features'); ylabel('Weight');

% Define minkowski function that takes two vectors or matrices
% and the parameter p and returns the distance or vector of distances
% between the examples.
function d = minkowski_dist(a, b, p)
	d = sum(abs(a - b).^p, 2).^(1/p);
end