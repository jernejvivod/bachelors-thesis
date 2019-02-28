function [r] = min_radius(n, data, target) 
	min_r = zeros(size(data, 1), 1);
	
	dist_mat = squareform(pdist(data, 'minkowski', 2));
	
	for k = 1:size(data, 1)
		dist_from_e = dist_mat(k, :);
		msk = target == target(k);
		dist_same = dist_from_e(msk);
		dist_diff = dist_from_e(~msk);
		
		min_list1 = sort(dist_same); min_list2 = sort(dist_diff);
		min_r(k) = max(min_list1(n+1), min_list2(n));
	end
	
	r = max(min_r);
	
end