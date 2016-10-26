function [ w ] = logistic_regression_training( x, y, iter, a, reg)
% Regressão Logística
% Entrada: x (dados de treinamento), y (rótulos), 

x = horzcat(ones(size(x,1),1), x);
w = unifrnd(-1,1,size(x,2),1);

for i=0:iter
    grad = gradient_error(x, y, w, reg);    
    w = w - a.*grad;
    w = w./norm(w);
    if(abs(sum(grad)) < 0.0001)
        break;
    end
end

end

