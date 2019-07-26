res1 = zeros(1, 300);
res2 = zeros(1, 300);

for k = 1:300
    disp(k)
    data = rand(1000, k);
    res1(k) = mean(pdist(data, 'Euclidean'));
    res2(k) = mean(pdist(data, 'Cityblock'));
end
