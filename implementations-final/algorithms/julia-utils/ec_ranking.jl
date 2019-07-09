using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn
using Statistics
@sk_import naive_bayes: GaussianNB

function ec_ranking(data, target, weights, mu_vals)

	# Get maximal ReliefF weight and compute epsilon values.
	max_weight = maximum(weights)
	eps = (max_weight .- weights)./max_weight


	# Set initial tinfo value.
	tinfo = 1


	# Initialize vector of feature ranks.
	rank = Array{Int64}(undef, length(weights))

	# Initialize vector of indices in original weights array.
	index_vector = collect(1:length(weights))
	
	# Initialize initial rank value.
	rank_value_next = length(weights)
	
	# Initialize variable that holds current best tinfo value.
	best_tinfo = tinfo

	# While there are unranked features, perform evaporation.
	while length(index_vector) > 1
		

		### Grid search for best temperature from previous step ###


		# Initialize variables that hold current maximal CV score 
		# and index of removed feature.
		max_cv_val = 0.0
		idx_removed_feature = -1

		
		# Perform grid search for best value of tinfo.
		for tinfo_nxt = tinfo-0.3:0.1:tinfo+0.3

			# Compute weights for next value of tinfo.
			weights_nxt = eps .- tinfo_nxt.*mu_vals

			# Rank features.
			enumerated_weights = [weights_nxt collect(1:length(weights_nxt))]'
			rank_weights = zeros(Int64, length(weights_nxt))
			s = enumerated_weights[:, sortperm(enumerated_weights[1,:], rev=false)]
			rank_weights[convert.(Int64, s[2, :])] = 1:length(weights_nxt)


			# Remove lowest ranked feature.
			msk_rm = rank_weights .!= length(weights_nxt)

			# Perform 5-fold cross validation.
			data_filt = data[:, msk_rm]

			# Compare to current maximal CV value.
			cv_val_nxt = Statistics.mean(cross_val_score(GaussianNB(), data_filt, target; cv=5))

			# If CV value greater than current maximal, save current
			# CV score, tinfo and feature weights.
			if cv_val_nxt > max_cv_val
				# Set current tinfo value as best.
				best_tinfo = tinfo_nxt
				# Get index of removed feature.
				idx_removed_feature = findall(.!msk_rm)[1]
			end
		end
	
		# Remove evaporated feature from data matrix,
		# and data at corresponding index from eps vector and 
		# mu values vector.
		data = data[:, 1:end .!= idx_removed_feature]
		eps = eps[1:end .!= idx_removed_feature]
		mu_vals = mu_vals[1:end .!= idx_removed_feature]


		# Get index of evaporated feature in original data matrix.
		rank_index_next = index_vector[idx_removed_feature]

		# Delete removed feature from vector of indices.
		index_vector = deleteat!(index_vector, idx_removed_feature)

		# Assign rank to removed feature.
		rank[rank_index_next] = rank_value_next

		# Decrease next rank value.
		rank_value_next -= 1

	end
	
	# Assign final rank.
	rank[index_vector[1]] = rank_value_next

	# Return vector of ranks.
	return rank

end

