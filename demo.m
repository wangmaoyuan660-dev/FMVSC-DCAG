clear;
clc;

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir, 'measure'));

data = load(fullfile(rootDir, 'data', '3Sources.mat'));
X = data.X;
labels = data.y(:);

numClusters = numel(unique(labels));
numAnchors = 12;
subspaceDim = 12;
beta = 0.0625;
lambda = 0.125;

[~, G0] = inG(numClusters, 150, 1e-5, numAnchors);

fprintf('Running FMVSC-DCAG on 3Sources...\n');
tic;
[embedding, ~, ~, ~, ~, ~, ~, ~] = ...
    FMVSC_DCAG(X, labels, lambda, subspaceDim, numAnchors, G0, beta);
elapsedTime = toc;

metrics = myNMIACCwithmean(embedding, labels, numClusters);
fprintf('ACC: %.4f\n', metrics(1));
fprintf('NMI: %.4f\n', metrics(2));
fprintf('Purity: %.4f\n', metrics(3));
fprintf('Time: %.2f seconds\n', elapsedTime);
