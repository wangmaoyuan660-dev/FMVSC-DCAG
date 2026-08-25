function [U, A, P, Z, iter, obj, alpha, G] = ...
    FMVSC_DCAG(X, labels, lambda, d, m, G0, beta)

maxIter = 350;
tol = 1e-5;
c = length(unique(labels));
numview = length(X);
n = size(labels, 1);

P = cell(numview, 1);
A = randn(d, m);
Z = zeros(m, n);
Z(:, 1:m) = eye(m);
G = G0;
alpha = ones(1, numview) / numview;

qpOptions = optimset('Algorithm', 'interior-point-convex', 'Display', 'off');
AeqZ = ones(1, m);
beqZ = 1;
lbZ = zeros(m, 1);
ubZ = ones(m, 1);

for v = 1:numview
    X{v} = mapstd(X{v}', 0, 1);
    P{v} = zeros(size(X{v}, 1), d);
end

iter = 0;
obj = zeros(maxIter, 1);
flag = true;

while flag
    iter = iter + 1;

    AZ = A * Z;
    parfor v = 1:numview
        C = X{v} * AZ';
        [Utmp, ~, Vtmp] = svd(C, 'econ');
        P{v} = Utmp * Vtmp';
    end

    b = trace(A * A');
    S = sum(alpha.^2);
    B = zeros(d, m);
    for v = 1:numview
        B = B + alpha(v)^2 * (P{v}' * X{v} * Z');
    end
    G_inv = G * pinv(G' * G);
    F_fixed = A * G_inv;
    B = B + (beta / b) * (F_fixed * G');
    R = S * (Z * Z') + (beta / b) * eye(m);
    A = B / R;

    H = lambda * eye(m);
    E = zeros(m, n);
    for v = 1:numview
        H = H + alpha(v)^2 * (A' * A);
        E = E + alpha(v)^2 * (A' * P{v}' * X{v});
    end
    QZ = 2 * H;
    QZ = (QZ + QZ') / 2;
    Z_qp = zeros(m, n);
    parfor j = 1:n
        fZ = -2 * E(:, j);
        Z_qp(:, j) = quadprog(QZ, fZ, [], [], ...
            AeqZ, beqZ, lbZ, ubZ, [], qpOptions);
    end
    Z = Z_qp;

    F = A * G_inv;
    I = eye(c);
    for j = 1:m
        dist = zeros(c, 1);
        for k = 1:c
            dist(k) = beta * norm(A(:, j) - F(:, k))^2;
        end
        [~, idx] = min(dist);
        G(j, :) = I(idx, :);
    end

    M = zeros(numview, 1);
    for v = 1:numview
        M(v) = norm(X{v} - P{v} * A * Z, 'fro')^2;
    end
    alpha = (M.^(-1)) / sum(M.^(-1));

    term1 = 0;
    for v = 1:numview
        term1 = term1 + alpha(v)^2 * norm(X{v} - P{v} * A * Z, 'fro')^2;
    end
    term2 = lambda * norm(Z, 'fro')^2;
    term3 = beta * norm(A - F * G', 'fro')^2 / trace(A * A');
    obj(iter) = term1 + term2 + term3;

    if (iter > 2) && ...
            (abs((obj(iter - 1) - obj(iter)) / obj(iter - 1)) < tol || ...
            iter >= maxIter)
        flag = false;
    end
end

obj = obj(1:iter);
[U, ~, ~] = mySVD(Z', c);
end
