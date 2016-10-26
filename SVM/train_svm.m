function [alpha, b, SV] = train_svm(X, y, C)

	n = size(X, 1);
	m = size(X, 2);

	K = X * X';

	H = (y * y') .* (K + (1/C) * eye(n));

	f = - ones(1, n);
	A = zeros(1, n);
	c = 0;
	Aeq = y';
	ceq = 0;
	LB = zeros(n, 1);
	UB = inf * ones(n, 1);

	alpha = quadprog(H,f,A,c,Aeq,ceq,LB,UB);

	SV = find(alpha > 0.0001);

	b = mean( 1 ./ y(SV) - K(SV, :) * (alpha .* y) );

end
