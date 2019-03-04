dataset = load('rba_test_data3.mat');
target = dataset.dataset.target; data = dataset.dataset.data;

lim_noise = 10;
noisy_data =  [data, rand(size(data, 1), lim_noise)];

res = zeros(lim_noise, 4);

figure; hold on; title(sprintf('ReliefF Weights for Monotonic and Non-monotonic Relevant\nFeature with Respect to Number of Noise Features'));	
% l1 = animatedline('Color','b');
xlim([0, lim_noise]); ylim([-0.2, 1]); xlabel('number of noise features'); ylabel('weight');
pause(0.5);
for k = 0:lim_noise
	[~, weights] = relieff_animation([noisy_data(:, 1:2+k), target], size(data, 1), 3, @(a, b) minkowski_dist(a, b, 2), 0);
	res(k+1, 1) = weights(1);
	res(k+1, 2) = weights(2);
	if k > 0
		res(k+1, 3) = mean(weights(3:end));
		res(k+1, 4) = max(weights(3:end));
	else
		res(k+1, 3) = NaN;
		res(k+1, 4) = NaN;
	end
	
	% addpoints(l1, k, res(k+1, 1));
	
	%plot(k, res(k+1, 1), 'b-'); plot(k, res(k+1, 2), 'g.');
	plot(k, res(k+1, 3), 'r.'); plot(k, res(k+1, 4), 'y.');
	t1 = text(k, res(k+1, 1) + 0.05, 'monotonic relevant feature weight');
	t2 = text(k, res(k+1, 2) + 0.05, 'non-monotonic relevant feature weight');
	t3 = text(k, res(k+1, 3) + 0.05, 'average noise feature weight');
	t4 = text(k, res(k+1, 3) + 0.05, 'maximal noise feature weight');
	pause(0.1);
	delete(t1); delete(t2); delete(t3); delete(t4);
end

figure; title(sprintf('ReliefF Weights for Monotonic and Non-monotonic Relevant\nFeature with Respect to Number of Noise Features'));
hold on; xlabel('number of noise features'); ylabel('weight');
plot(0:lim_noise, res(:, 1), 'b-');
plot(0:lim_noise, res(:, 2), 'g-');
plot(0:lim_noise, res(:, 3), 'r-');
plot(0:lim_noise, res(:, 4), 'y-');

legend('monotonic feature weight',...
	'non-monotonic feature weight',...
	'average noise features weight',...
	'maximal noise features weight',...
	'Location', 'northeast');

xlim([0, lim_noise]); ylim([-0.2, 1]);

% Define minkowski function that takes two vectors or matrices
% and the parameter p and returns the distance or vector of distances
% between the examples.
function d = minkowski_dist(a, b, p)
	d = sum(abs(a - b).^p, 2).^(1/p);
end