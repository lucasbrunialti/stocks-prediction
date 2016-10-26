function [X_norm] = reduce_dim(X)

    avg = mean(X);
    X = X - repmat(avg, size(X, 1), 1) ./ repmat(std(X), size(X, 1), 1);
    % sigma = X * X' ./ size(X, 2);

    % [U,S,V] = svd(sigma);

    % X_norm = diag(1 ./ sqrt(diag(S))) * U' * X;

    X_norm = (X - repmat(min(X), size(X,1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);
end
