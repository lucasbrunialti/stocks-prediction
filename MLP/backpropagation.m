function [ W1_best, W2_best, B1_best, B2_best, mse_train, mse_test ] = backpropagation(X_train, expected_train, X_test, expected_test, num_epochs, num_neurons_hid, learning_rate, max_momentum_rate, learning_rate_incr, learning_rate_dec, regularization_rate)

    % Reference: S. Haykin, Neural Networks: A Comprehensive Foundation,
    % 2nd Edition, Prentice-Hall, 1999, Coursera Machine Learning Class
    % Notes (Stanford University) - https://www.coursera.org/course/ml and
    % article APRENDIZADO POR TRANSFERENCIA PARA APLICACOES ORIENTADAS
    % A USUARIO: UMA EXPERIENCIA EM LINGUA DE SINAIS
    %
    % Note that the implementation presented here is modified from the
    % original reference.

    m = size(X_train, 2);

    num_neurons_output = size(expected_train, 2);

    % heuristic
    a = - 4 * sqrt(6 / (num_neurons_hid + m));
    b = 4 * sqrt(6 / (num_neurons_hid + m));

    % random initialization of weights
    W1 = rand(num_neurons_hid, m);
    W2 = rand(num_neurons_output, num_neurons_hid);
    W1 = a + (b-a) .* W1;
    W2 = a + (b-a) .* W2;

    % random initialization of bias weights
    B1 = a + (b-a) .* rand(num_neurons_hid, 1);
    B2 = a + (b-a) .* rand(num_neurons_output, 1);

    % store weights for each epoch
    W1_best = W1;
    W2_best = W2;
    B1_best = B1;
    B2_best = B2;

    % number of instances
    n_train = size(X_train,1);
    n_test = size(X_test,1);

    % mse for each epoch
    mse_train = ones(num_epochs,1)*Inf;
    mse_test = ones(num_epochs,1)*Inf;

    %figure;
    %hold on;
    %grid on;
    %xlabel('epochs');
    %ylabel('MSE');

    Grad_W2 = zeros(size(W2));
    Grad_W1 = zeros(size(W1));
    Grad_B2 = zeros(size(B2));
    Grad_B1 = zeros(size(B1));

    for i=1:num_epochs

        % update momentum
        momentum = min(1 - 2^(-1-log2(floor(i/250)+1)), max_momentum_rate);

        % store weights
        %W1_epochs(:,:,i) = W1;
        %W2_epochs(:,:,i) = W2;

        % randomly shuffle examples of the dataset, to create a Stochastic
        % Gradient Descent
        idx = randperm(size(X_train,1));
        X_train = X_train(idx, :);
        expected_train = expected_train(idx, :);

        % Batch gradient computation (matrix form)
        if (i == 1)
            % randomly shuffle examples of the dataset, to create a Stochastic
            % Gradient Descent
            idx = randperm(size(X_train,1));
            X_train = X_train(idx, :);
            expected_train = expected_train(idx, :);

            [A3_train, Z3, A2, Z2] = feedforward(X_train, W1, W2, B1, B2);
            A3_test = feedforward(X_test, W1, W2, B1, B2);


            error_train = - (double(expected_train) - A3_train);
            error_test = - (double(expected_test) - A3_test);

            % Summation over the output neuros
            mse_train(i) = (1/n_train) * sum(sum(error_train.^2));
            mse_test(i) = (1/n_test) * sum(sum(error_test.^2));


            Delta3 = error_train .* dsigmoid(Z3);
            Delta2 = (Delta3 * W2) .* dsigmoid(Z2);
            Grad_W2 = (1/n_train) * Delta3' * A2;
            Grad_W1 = (1/n_train) * Delta2' * X_train;
            Grad_B2 = mean(Delta3)';
            Grad_B1 = mean(Delta2)';
        else
            W1 = W1 + momentum * DeltaW1;
            W2 = W2 + momentum * DeltaW2;
            B1 = B1 + momentum * DeltaB1;
            B2 = B2 + momentum * DeltaB2;

            [A3_train, Z3, A2, Z2] = feedforward(X_train, W1, W2, B1, B2);
            A3_test = feedforward(X_test, W1, W2, B1, B2);


            error_train = - (double(expected_train) - A3_train);
            error_test = - (double(expected_test) - A3_test);

            % Summation over the output neuros
            mse_train(i) = (1/n_train) * sum(sum(error_train.^2));
            mse_test(i) = (1/n_test) * sum(sum(error_test.^2));


            Delta3 = error_train .* dsigmoid(Z3);
            Delta2 = (Delta3 * W2) .* dsigmoid(Z2);
            Grad_W2 = (1/n_train) * Delta3' * A2;
            Grad_W1 = (1/n_train) * Delta2' * X_train;
            Grad_B2 = mean(Delta3)';
            Grad_B1 = mean(Delta2)';

        end

        % Weights update
        if (i == 1)
            % DeltaW2 = - learning_rate * Grad_W2;
            % DeltaW1 = - learning_rate * Grad_W1;
            DeltaB2 = - learning_rate * Grad_B2;
            DeltaB1 = - learning_rate * Grad_B1;
            % Regularization
            DeltaW2 = - learning_rate * (Grad_W2 + regularization_rate * W2);
            DeltaW1 = - learning_rate * (Grad_W1 + regularization_rate * W1);
        else
            DeltaW2 = momentum * DeltaW2 - learning_rate * (Grad_W2 + regularization_rate * W2);
            DeltaW1 = momentum * DeltaW1 - learning_rate * (Grad_W1 + regularization_rate * W1);
            DeltaB2 = momentum * DeltaB2 - learning_rate * Grad_B2;
            DeltaB1 = momentum * DeltaB1 - learning_rate * Grad_B1;

            % adaptive learning rate
            if (mse_train(i) > mse_train(i-1))
                learning_rate = max(0.05, learning_rate * learning_rate_dec);
            else
                learning_rate = learning_rate * learning_rate_incr;
            end
        end

        W2 = W2 + DeltaW2;
        W1 = W1 + DeltaW1;

        B2 = B2 + DeltaB2;
        B1 = B1 + DeltaB1;

        if (min(mse_test) >= mse_test(i))
            W1_best = W1;
            W2_best = W2;
            B2_best = B2;
            B1_best = B1;
        end

        %if (i ~= 1)
        %    plot([i-1 i], [mse_train(i-1) mse_train(i)],'b');
        %    plot([i-1 i], [mse_test(i-1) mse_test(i)],'r');
        %    drawnow;
        %end

        %mean(abs(error_test))
        %mse_train(i)

    end

   % hold off;

end
