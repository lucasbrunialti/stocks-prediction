function [ a, b, tipo, beta ] = elm_training( x, y, neuronios, reg )

[n, m] = size(x);
nonsing = 0;

while(nonsing == 0)

    a = unifrnd(-1, 1, neuronios, m);
    b = unifrnd(-1, 1, neuronios, 1);
    tipo = randi([1 2], neuronios, 1);

    h = ones(n, neuronios + 1);

    for i = 1:n
        for j = 1:neuronios
            h(i, j) = ativacao(x(i,:), a(j, :), b(j), tipo(j));
        end
    end

    hi1 = h'*h;
    hi2 = h*h';
    if(reg ~= 0)
      hi1 = (eye(size(hi1)) ./ reg) + hi1;
      hi2 = (eye(size(hi2)) ./ reg) + hi2;
    end

    if(rcond(hi1) < 1e-12 && rcond(hi2) < 1e-12)
      disp('warning: singular')
      continue;
    else
      nonsing = 1;
    end

    if( rcond(hi1) > 1e-12 )
      pseudo_inv = inv(hi1)*h';
    else
      pseudo_inv = h'*inv(hi2);
    end

    beta = pseudo_inv*y;
end

end

