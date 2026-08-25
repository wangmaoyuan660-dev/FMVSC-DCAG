function [inPara, G0] = inG(numCluster, maxIter, thresh, m)
I = eye(numCluster);
order = randperm(numCluster);
copies = ceil(m / numCluster);
G0 = repmat(I(order, :), copies, 1);
G0 = G0(1:m, :);
inPara = struct('maxIter', maxIter, ...
                 'thresh', thresh, ...
                 'numCluster', numCluster);
end
