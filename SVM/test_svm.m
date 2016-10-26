function [F] = test_svm(Xtrain, ytrain, Xtest, alpha, b, SV)

	n = size(Xtest, 1);

	F = zeros(n, 1);

	for j= 1:n
		for i=1:size(SV,1)
			l = SV(i);
			F(j) = F(j) + ytrain(l) * alpha(l) * (Xtrain(l, :) * Xtest(j, :)');
		end
		F(j) = F(j) + b;
	end

end