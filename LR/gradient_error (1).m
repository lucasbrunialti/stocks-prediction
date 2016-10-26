function [ G ] = gradient_error( x, y, w, reg)
% Gradiente do erro da regressão logística.

n = size(x,1);
m = size(x,2);

partial = repmat(y, 1, m) .* x;
G = - mean( partial ./ (1 + exp( partial * repmat(w, 1, m) )) );
G = G + reg*[0 w(2:end)'];
G = G';


end

