function [inPara, G0] = inG(numCluster, maxIter, thresh, m)

I = eye(numCluster);
R = randperm(numCluster);
g = ceil(m / numCluster);
inG0 = repmat(I(R, :), g, 1);
G0 = inG0(1:m, :);

inPara = struct('maxIter', maxIter, ...
    'thresh', thresh, ...
    'numCluster', numCluster);
end
