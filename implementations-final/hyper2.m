res1 = zeros(1, 300);
res2 = zeros(1, 300);

for k = 1:300
    disp(k)
    data = rand(1000, k);
    dm1 = squareform(pdist(data, 'Euclidean'));
    dm2 = squareform(pdist(data, 'Cityblock'));
    
    ratios1 = zeros(1, size(dm1, 1));
    ratios2 = zeros(1, size(dm1, 1));
    for l = 1:size(dm1, 1)
        min1 = mink(dm1(l, :), 2);
        min1 = min1(2);
        min2 = mink(dm2(l, :), 2);
        min2 = min2(2);
        
        max1 = max(dm1(l, :));
        max2 = max(dm2(l, :));
        
        ratios1(l) = min1/max1;
        ratios2(l) = min2/max2;
        
    end
    res1(k) = mean(ratios1);
    res2(k) = mean(ratios2);
    
end
