relief <- function(data, target, m, dist_func) {
	#' Perform feature ranking using Relief algorithm
	#' 
	#' @description This function performs feature ranking using
	#' Relief algorithm.
	#' 
	#' @param data - matrix containing features of examples
	#' @param target - vector containing classes of examples
	#' @param m - size of sample of examples. Default value is the total number of examples.
	#' @usage relief(data, target, m, dist_func)
	#' @return vector of feature weights/scores
	#' @export
	#' @importFrom base paste


	# Initialize weight of each feature to 0.
	weights <- replicate(ncol(data), 0)

	# Get maximum and minimum values of each feature.
	max_f_vals <- apply(data, 2, max)
	min_f_vals <- apply(data, 2, min)

	# Get indices of examples in the sample.
	sample_idxs <- sample(nrow(data), m, replace=FALSE)

	# Go over indices of sampled examples.
	for (idx in sample_idxs) {

		# Get sampled example data.
		e <- data[idx, ]

		# Get mask for examples from same class.
		msk <- target == target[idx]

		# Compute distances to exampels of same class and different class.
		dist_same <- apply(data[msk,], 1, function(row) { dist_func(e, row) })
		dist_other <- apply(data[!msk,], 1, function(row) { dist_func(e, row) })

		# Get index of selected example within elements from same class.
		# Set distance of sampled example to itself to infinity.
		idx_class <- idx - sum(target[1:idx-1] != target[idx])
		dist_same[[idx_class]] <- Inf

		# Get nearest hits and nearest misses.
		nearest_hit <- (data[msk,])[which.min(dist_same), ]
		nearest_miss <- (data[!msk,])[which.min(dist_other), ]

		# ------- Weights Update -------

		# Go over feature indices.
		for (t in 1:ncol(data)) {
			# Update weights.
			weights[t] <- weights[t] - (abs(e[[t]] - nearest_hit[[t]])/(max_f_vals[[t]] - min_f_vals[[t]]))/m + (abs(e[[t]] - nearest_miss[[t]])/(max_f_vals[[t]] - min_f_vals[[t]]))/m
		}
	}
	
	# Rank features according to weights.
	names(weights) <- order(weights, decreasing=TRUE)
	return(weights)
}