% Test effects of adding noise features on weight of relevant features.

dataset = load('simple_monotonic.mat');
target = dataset.dataset.target; data = dataset.dataset.data;

lim_noise = 100;
noisy_data =  [data, rand(size(data, 1), lim_noise)];

res = zeros(lim_noise, 3);

shift_val = 100;

figure; hold on; title(sprintf('Iterative Relief Weights for Monotonic and Non-monotonic\nRelevant Feature with Respect to Number of Noise Features'));

l1 = animatedline('Color','b');
l2 = animatedline('Color','r');
l3 = animatedline('Color','y');

xlim([0, lim_noise]); ylim([-0.05, 0.5]); xlabel('number of noise features'); ylabel('weight');
pause(0.5);
for k = 0:lim_noise
	[~, weights] = relieff_animation([noisy_data(:, 1:1+k), target], size(data, 1), 3, @(a, b) minkowski_dist(a, b, 2), 0);
	res(k+1, 1) = weights(1);
	if k > 0
		res(k+1, 2) = mean(weights(2:end));
		res(k+1, 3) = max(weights(2:end));
	else
		res(k+1, 2) = NaN;
		res(k+1, 3) = NaN;
	end
	
		
	addpoints(l1, k, res(k+1, 1));
	addpoints(l2, k, res(k+1, 2));
	addpoints(l3, k, res(k+1, 3));
	
	t1 = text(k, res(k+1, 1) + 0.02, 'monotonic relevant feature weight');
	t2 = text(k, res(k+1, 2) + 0.02, 'average noise feature weight');
	t3 = text(k, res(k+1, 3) + 0.02, 'maximum noise feature weight');
	
	t4 = text(5, 280, sprintf('r1 = %.2f', (res(k+1, 1) + shift_val)/(res(k+1, 2) + shift_val)));
	
	pause(0.1);
	delete(t1); delete(t2); delete(t3); delete(t4);
end

figure; title(sprintf('Iterative Relief Weights for Monotonic and Non-monotonic Feature\nwith Respect to Number of Noise Features'));	
hold on; xlabel('Number of Noise Features'); ylabel('Weight');
plot(0:lim_noise, res(:, 1), 'b-');
plot(0:lim_noise, res(:, 2), 'r-');
plot(0:lim_noise, res(:, 3), 'y-');

legend('monotonic relevant feature weight',...
	'average noise features weight',...
	'maximal noise feature weight',...
	'Location', 'southeast');

xlim([0, lim_noise]); ylim([-0.05, 0.35]); xlabel('Number of Noise Features'); ylabel('Weight');


% Define minkowski function that takes two vectors or matrices
% and the parameter p and returns the distance or vector of distances
% between the examples.
function d = minkowski_dist(a, b, p)
	d = sum(abs(a - b).^p, 2).^(1/p);
end