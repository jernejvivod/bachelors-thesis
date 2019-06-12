data = np.array([[2.09525, 0.26961, 3.99627],
                 [9.86248, 6.22487, 8.77424],
                 [7.03015, 9.24269, 3.02136],
                 [8.95009, 8.52854, 0.16166],
                 [3.41438, 4.03548, 7.88157],
                 [2.01185, 0.84564, 6.16909],
                 [2.79316, 1.71541, 2.97578],
                 [3.22177, 0.16564, 5.79036],
                 [1.81406, 2.74643, 2.13259],
                 [4.77481, 8.01036, 7.57880]])
target = np.array([1, 2, 2, 2, 1, 1, 3, 3, 3, 1])

dist = lambda x1, x2 : np.sum(np.abs(x1-x2), 1)

distances = dist(e, data)
print(distances)
print(target)

idx_closest_same1 = int(input("closest same 1"))
idx_closest_same2 = int(input("closest same 2"))
idx_closest_other11 = int(input("closest other 11"))
idx_closest_other12 = int(input("closest other 12"))
idx_closest_other31 = int(input("closest other 31"))
idx_closest_other32 = int(input("closest other 32"))

nearest_hit1 = data[idx_closest_same1, :]
nearest_hit2 = data[idx_closest_same2, :]

nearest_miss11 = data[idx_closest_other11]
nearest_miss12 = data[idx_closest_other12]
nearest_miss31 = data[idx_closest_other31]
nearest_miss32 = data[idx_closest_other31]

diff = lambda idx_feature, x1, x2: (1/ranges[idx_feature])*np.abs(x1[0], x2[0])
reward_term = lambda idx_feature, pc, pr, m, k, e, miss1, miss2: (pc/(1-pr)) * (diff(idx_feature, e, miss1) + diff(idx_feature, e, miss2))

weight_update1 = lambda idx_feature, weight : weight - diff(idx_feature, e, nearest_hit1) - diff(idx_feature, e, nearest_hit2) + reward_term(idx_feature, 0.4, 0.7, m, k, e, nearest_miss11, nearest_miss12)
weight_update2 = lambda idx_feature, weight : weight - diff(idx_feature, e, nearest_hit1) - diff(idx_feature, e, nearest_hit2) + reward_term(idx_feature, 0.4, 0.7, m, k, e, nearest_miss11, nearest_miss12)

weights_mult = np.array([4.0/10.0, 3.0/10.0, 3.0/10.0])

