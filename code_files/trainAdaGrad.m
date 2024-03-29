function [hiddenWeights, outputWeights, error] = trainAdaGrad(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
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
    
    % AdaGrad terms
    % G terms are used for accumulating the gradients
    G_1 = rand(numberOfHiddenUnits, inputDimensions);
    G_1_next = rand(numberOfHiddenUnits, inputDimensions);
    G_2 = rand(outputDimensions, numberOfHiddenUnits);
    G_2_next = rand(outputDimensions, numberOfHiddenUnits);
    
    % Initializing epsilon terms to avoid "division by zero" problems
    g1_dims = size(G_1);
    g2_dims = size(G_2);    
    eps_1 = repmat(1/(g1_dims(1)*g1_dims(2)), size(G_1));
    eps_2 = repmat(1/(g2_dims(1)*g2_dims(2)), size(G_2));
    
    % Initializing weights
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    
    figure; hold on;

    for t = 1: epochs
        for k = 1: batchSize
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
            
            
            g_ow = outputDelta*hiddenOutputVector';
            % Accumulating gradients
            G_2_next = G_2 + g_ow.^2;
            % Scaling the learning rate by using 
            % the previous cumulative gradient term
            outputWeights = outputWeights - learningRate./sqrt(G_2 + eps_2) .* g_ow;
            
            g_hw = hiddenDelta*inputVector';
            % Accumulating gradients
            G_1_next = G_1 + g_hw.^2;
            % Scaling the learning rate by using 
            % the previous cumulative gradient term
            hiddenWeights = hiddenWeights - learningRate./sqrt(G_1 + eps_1) .* g_hw;
            
            G_1 = G_1_next;
            G_2 = G_2_next;
                    
        end;      
        
        disp(t);
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