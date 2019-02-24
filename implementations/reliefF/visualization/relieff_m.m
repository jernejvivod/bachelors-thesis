function [ranked,weight] = relieff_m(X, Y, k, varargin)

% Find max and min for every feature
num_features = size(X, 2);  % number of features
Xmax = max(X);  % Smallest values of each feature
Xmin = min(X);  % Largest value of each feature
Xdiff = Xmax-Xmin;  % Difference between largest and smallest value of each feature

% Scale and center the features.
X = bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),Xdiff);

% Get appropriate distance function in one dimension.
% thisx must be a row-vector for one observation.
% x can have more than one row.
dist1D = @(thisx,x) cityblock(thisx,x);

% Call ReliefF. Initialize all weights with NaN
weight = NaN(1, num_features);
weight(accepted) = relieffClass(X, Y, classProb, numUpdates, k, distFcn, dist1D, sigma);


% -------------------------------------------------------------------------
function attrWeights = relieffClass(scaledX, C, classProb, m, k, dist_func, dist1D, sigma)
% ReliefF for classification

[num_examples, num_features] = size(scaledX);
attrWeights = zeros(1, num_features);
Nlev = size(C,2);

% Choose the random instances
sample_idxs = randsample(num_examples, m);
idxVec = (1:num_examples)';

% Make searcher objects, one object per class. 
searchers = cell(Nlev,1);
for c = 1:Nlev
    searchers{c} = createns(scaledX(C(:, c), :), 'Distance', dist_func);
end

% Outer loop, for updating attribute weights iteratively
for i = 1:m
    idx_nxt_example = sample_idxs(i);
    
    % Choose the correct random observation
    nxt_example = scaledX(idx_nxt_example, :);

    % Find the class for this observation
    this_class = C(idx_nxt_example, :);
    
    % Find the k-nearest hits 
    sameClassIdx = idxVec(C(:, this_class));  % Indices of examples with same class
    
    % we may not always find numNeighbor Hits
    len_Hits = min(length(sameClassIdx) - 1, k);

    % find nearest hits
    % It is not guaranteed that the first hit is the same as thisObs. Since
    % they have the same class, it does not matter. If we add observation
    % weights in the future, we will need here something similar to what we
    % do in ReliefReg.
    Hits = [];
    if len_Hits > 0
        idxH = knnsearch(searchers{this_class}, nxt_example, 'K', len_Hits + 1);
        idxH(1) = [];  % Remove distance to self.
        Hits = sameClassIdx(idxH);
    end    
    
    % Process misses
    missClass = find(~this_class);
    Misses = [];
    
    if ~isempty(missClass) % Make sure there are misses!
        % Find the k-nearest misses Misses(C,:) for each class C ~= class(selectedX)
        % Misses will be of size (no. of classes -1)x(K)
        Misses = zeros(Nlev-1,min(num_examples,k+1)); % last column has class index
        
        for mi = 1:length(missClass)
            
            % find all observations of this miss class
            missClassIdx = idxVec(C(:,missClass(mi)));
            
            % we may not always find K misses
            lenMiss = min(length(missClassIdx),k);
            
            % find nearest misses
            idxM = knnsearch(searchers{missClass(mi)},nxt_example,'K',lenMiss);
            Misses(mi,1:lenMiss) = missClassIdx(idxM);
            
        end
        
        % Misses contains obs indices for miss classes, sorted by dist.
        Misses(:,end) = missClass;
    end
            
    %***************** ATTRIBUTE UPDATE *****************************
    % Inner loop to update weights for each attribute:
    
    for j = 1:num_features
        dH = diffH(j,scaledX,idx_nxt_example,Hits,dist1D,sigma)/m;
        dM = diffM(j,scaledX,idx_nxt_example,Misses,dist1D,sigma,classProb)/m;
        attrWeights(j) = attrWeights(j) - dH + dM;
    end
    %****************************************************************
end


%Helper functions for RelieffClass

%--------------------------------------------------------------------------
% DIFFH (for RelieffClass): Function to calculate difference measure
% for an attribute between the selected instance and its hits

function distMeas = diffH(a,X,thisObs,Hits,dist1D,sigma)

% If no hits, return zero by default
if isempty(Hits)
    distMeas = 0;
    return;
end

% Get distance weights
distWts = exp(-((1:length(Hits))/sigma).^2)';
distWts = distWts/sum(distWts);

% Calculate weighted sum of distances
distMeas = sum(dist1D(X(thisObs,a),X(Hits,a)).*distWts);


%--------------------------------------------------------------------------
% DIFFM (for RelieffClass) : Function to calculate difference measure
% for an attribute between the selected instance and its misses
function distMeas = diffM(a,X,thisObs,Misses,dist1D,sigma,classProb)

distMeas = 0;

% If no misses, return zero
if isempty(Misses)
    return;
end

% Loop over misses
for mi = 1:size(Misses,1)
    
    ismiss = Misses(mi,1:end-1)~=0;
    
    if any(ismiss)
        cls = Misses(mi,end);
        nmiss = sum(ismiss);
        
        distWts = exp(-((1:nmiss)/sigma).^2)';
        distWts = distWts/sum(distWts);
        
        distMeas = distMeas + ...
            sum(dist1D(X(thisObs,a),X(Misses(mi,ismiss),a)).*distWts(1:nmiss)) ...
            *classProb(cls);
    end
end

% Normalize class probabilities.
% This is equivalent to P(C)/(1-P(class(R))) in ReliefF paper.
totProb = sum(classProb(Misses(:,end)));
distMeas = distMeas/totProb;

function d = cityblock(thisX,X)
d = abs(thisX-X);