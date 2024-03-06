addpath(genpath('./minFunc_2012'));


% Load MNIST.
X = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% Transform the labels to correct target values.
Y = 0.*ones(10, size(labels, 1));

for n = 1: size(labels, 1)
    Y(labels(n) + 1, n) = 1;
end;

X = X';
Y = Y';

[n, m] = size(Y);
[~, b] = size(X);

% W1 = rand(b, numberOfHiddenUnits);
% W2 = rand(numberOfHiddenUnits, m);
numberOfHiddenUnits = 10;
W = rand(b+m, numberOfHiddenUnits);

maxFunEvals = 100;

fun = @(w)loss_func(Y, X, w, numberOfHiddenUnits);
activationFunction = @logisticSigmoid;

options = [];
options.display = 'none';
options.useMex = 0; % For fair comparison in time
options.maxFunEvals = maxFunEvals;

%% Conjugate gradient
options.Method = 'cg'; 
[cg_x, cg_f, ~, cg_output] = minFunc(fun, W, options);
fprintf('Conjugate Gradient Objective Function Value: %f\n', cg_f);

hiddenWeights = cg_x(1:b, :)';
outputWeights = cg_x(b+1:b+m, :);

inputValues = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Choose decision rule.
fprintf('Validation:\n');
    
[correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels);
    
fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);