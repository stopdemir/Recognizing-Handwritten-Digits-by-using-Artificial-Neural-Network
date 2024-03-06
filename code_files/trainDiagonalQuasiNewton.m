function [hiddenWeights, outputWeights, error] = trainDiagonalQuasiNewton(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    % Initializing epsilon terms to avoid "division by zero" problems
    g1_dims = size(hiddenWeights);
    g2_dims = size(outputWeights);    
    eps_1 = 0.000000001 .* repmat(1/(g1_dims(1)*g1_dims(2)), size(hiddenWeights));
    eps_2 = 0.000000001 .* repmat(1/(g2_dims(1)*g2_dims(2)), size(outputWeights));
    
    a_1 = 1;
    a_2 = 1;
    
    n = zeros(batchSize);
    
    figure; hold on;

    for t = 1: epochs
        for k = 1: batchSize
            ow_x = ones([size(outputWeights), 2]);
            hw_x = ones([size(hiddenWeights), 2]);
            ow_g = ones([size(outputWeights), 2]);
            hw_g = ones([size(hiddenWeights), 2]);
            for i = 1:2
                % Select which input vector to train on.
                n(k) = floor(rand(1)*trainingSetSize + 1);
            
                % Propagate the input vector through the network.
                inputVector = inputValues(:, n(k));
                hiddenActualInput = hiddenWeights*inputVector;
                hiddenOutputVector = activationFunction(hiddenActualInput);
                outputActualInput = outputWeights*hiddenOutputVector;
                outputVector = activationFunction(outputActualInput);
            
                targetVector = targetValues(:, n(k));
            
                % Backpropagate the errors.
                outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
                hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
                ow_g(:, :, i) = outputDelta*hiddenOutputVector';
                hw_g(:, :, i) = hiddenDelta*inputVector';
                ow_x(:, :, i) = ow_x(:, :, i) - learningRate.*ow_g(:, :, i);
                hw_x(:, :, i) = hw_x(:, :, i) - learningRate.*hw_g(:, :, i);
            end
            
                        
            % S_k+1 = X_k+1 - X_k
            ow_s = ow_x(:, :, 2) - ow_x(:, :, 1);
            hw_s = hw_x(:, :, 2) - hw_x(:, :, 1);
            
            % Y_k+1 = g_k+1 - g_k
            ow_y = ow_g(:, :, 2) - ow_g(:, :, 1);
            hw_y = hw_g(:, :, 2) - hw_g(:, :, 1);
            
            % Applying secant rule to approximate Hessian inverse (B_inv)
            % Computing the search direction d = B_inv * g
            Binv_ow = ow_s ./ (ow_y + eps_2);
            Binv_hw = hw_s ./ (hw_y + eps_1);
            
            Binv_ow = a_2 .* diag(sum(Binv_ow, 2));
            Binv_hw = a_1 .* diag(sum(Binv_hw, 2));
            
            ow_d = Binv_ow * ow_g(:, :, 1);
            hw_d = Binv_hw * hw_g(:, :, 1);
            
            outputWeights = outputWeights + learningRate.*ow_d;
            hiddenWeights = hiddenWeights + learningRate.*hw_d;
            
        end;
        disp(t);
        
        a_2 = norm(ow_g(:, :, 1), 2);
        a_1 = norm(hw_g(:, :, 1), 2);
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);
        end;
        error = error/batchSize;
        
        plot(t, error,'k*'); 
        xlabel('epoch');
        ylabel('error');
    end;
end