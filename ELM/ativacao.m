function [ result ] = ativacao( x, a, b, tipo )

switch tipo
    case 1
        result = 1./(1+exp(-(a*x'+b)));
    case 2
        if(a*x' - b >= 0)
            result = 1;
        else
            result = 0;
        end
    case 3
        result = exp(-b*norm(x - a)^2);
    case 4
        result = -(norm(x - a)^2 + b^2)^(1/2);
end


end

