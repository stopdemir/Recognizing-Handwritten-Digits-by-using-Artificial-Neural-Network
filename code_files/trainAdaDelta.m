function [hiddenWeights, outputWeights, error] = trainAdaDelta(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
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
    
    % AdaDelta terms
    % G terms will be used for accumulating gradients
    % deltaX terms will be used for accumulating updates
    G_1 = rand(numberOfHiddenUnits, inputDimensions);
    G_1_next = rand(numberOfHiddenUnits, inputDimensions);
    deltaX_1 = rand(numberOfHiddenUnits, inputDimensions);
    deltaX_1_next = rand(numberOfHiddenUnits, inputDimensions);
    G_2 = rand(outputDimensions, numberOfHiddenUnits);
    G_2_next = rand(outputDimensions, numberOfHiddenUnits);
    deltaX_2 = rand(outputDimensions, numberOfHiddenUnits);
    deltaX_2_next = rand(outputDimensions, numberOfHiddenUnits);
    rho = 0.6;
    
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
            % Accumulating gradient in AdaGrad-like fashion
            G_2_next = sqrt(rho .* G_2.^2 + (1 - rho) .* g_ow.^2);
            % Computing the update for output weights
            update_2 = - g_ow .* sqrt((deltaX_2.^2 + eps_2)./(g_ow.^2 + eps_2));
            % Accumulating updates in momentum-like fashion
            deltaX_2_next = sqrt(rho .* deltaX_2.^2 + (1 - rho) .* update_2.^2);
            outputWeights = outputWeights + update_2;
                       
            g_hw = hiddenDelta*inputVector';
            % Accumulating gradient in AdaGrad-like fashion
            G_1_next = sqrt(rho .* G_1.^2 + (1 - rho) .* g_hw.^2);
            % Computing the update for hidden weights
            update_1 = - g_hw .* sqrt((deltaX_1.^2 + eps_1)./(g_hw.^2 + eps_1));
            % Accumulating updates in momentum-like fashion
            deltaX_1_next = sqrt(rho .* deltaX_1.^2 + (1 - rho) .* update_1.^2);
            hiddenWeights = hiddenWeights + update_1;
            
            G_1 = G_1_next;
            G_2 = G_2_next;
            deltaX_1 = deltaX_1_next;
            deltaX_2 = deltaX_2_next;
                    
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