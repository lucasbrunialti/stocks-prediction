function [X_norm] = reduce_dim(X)

    avg = mean(X);
    X = (X - repmat(avg, size(X, 1), 1)) ./ repmat(std(X), size(X, 1), 1);

    X_norm = (X - repmat(min(X), size(X,1), 1)) ./ repmat(max(X) - min(X), size(X, 1), 1);
end
