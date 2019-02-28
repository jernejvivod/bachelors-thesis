function d = minkowski_dist_weighted(a, b, w, p)
	d = sum(w .* (abs(a - b).^p), 2).^(1/p);
end