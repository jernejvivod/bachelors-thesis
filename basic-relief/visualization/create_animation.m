function [] = create_animation()

	function d = minkowski_dist(a, b, p)
		d = sum(abs(a - b).^p, 2).^(1/p);
	end

	data1 = [0 0 0 0; 0 0 1 0; 0 1 1 1; 0 1 0 1; 1 0 0 1; 1 0 1 1; 1 1 0 0; 1 1 0 0];
	data2 = load('rba_test_data.m');
	weights = relief_animation(data2, 100,  @(a, b) minkowski_dist(a, b, 2));
end