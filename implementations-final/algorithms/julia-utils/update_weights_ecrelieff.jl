using ScikitLearn.CrossValidation: cross_val_score
@sk_import naive_bayes: GaussianNB

function update_weights(data, e, closest_same, closest_other, weights, 
						weights_mult, m, k, max_f_vals, min_f_vals, mu_vals, tinfo)

	# Compute ReliefF weights.
	for t = 1:size(data, 2)

		# Penalty term
		penalty = sum(abs.(e[t] .- closest_same[:, t])/((max_f_vals[t] .- min_f_vals[t]) .+ 1e-10))

		# Reward term
		reward = sum(weights_mult .* (abs.(e[t] .- closest_other[:, t])/((max_f_vals[t] .- min_f_vals[t] .+ 1e-10))))

		# Weights update
		weights[t] = weights[t] - penalty/(m*k) + reward/(m*k)
	end

	
	# Get maximal ReliefF weight and compute epsilon values.
	max_weight = maximum(weights)
	eps = (max_weight .- weights)./max_weight

	### Grid search for best temperature from previous step ###
	
	best_tinfo = tinfo
	idx_removed_feature = -1
		
	for tinfo_nxt = tinfo-0.3:0.1:tinfo+0.3

		# Compute weights for next value of tinfo.
		weights_nxt = eps .- tinfo.*mu_vals

		# Rank features.
		enumerated_weights = [weights_nxt; collect(1:size(weights_nxt, 2))[:,:]']
		rank = zeros(Int64, size(weights_nxt, 2))
		s = enumerated_weights[:, sortperm(enumerated_weights[1,:], rev=true)]
		rank[convert.(Int64, s[2, :])] = 1:size(weights_nxt, 2)
	
		# Remove lowest ranked feature.
		msk_rm = rank .!= length(weights_nxt)


		# Perform 5-fold cross validation.
		data_filt = data[:, msk_rm]

		# Compare to current maximal CV value.
		cv_val_nxt = Statistics.mean(cross_val_score(GaussianNB(), X, y; cv=5))
		# If CV value greater than current maximal, save current
		# CV score, tinfo and feature weights.
		if cv_val_nxt > max_cv_val
			# Set current tinfo value as best.
			best_tinfo = tinfo_nxt
			# Get index of removed feature.
			idx_removed_feature = findall(msk_rm)[1][2]
		end
	end	

	###########################################################

	return idx_removed_feature, best_tinfo

end


# Todo: Assign lowest rank to feature that is evaporated in next step.
#
