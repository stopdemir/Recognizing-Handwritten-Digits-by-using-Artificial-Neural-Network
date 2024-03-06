% Here we briefly mention purposes of the code files

Files of the first order methods: 
	trainAdaDelta.m
	trainAdaGrad.m
	trainDiagonalQuasiNewton.m
	trainMomentumSGD.m
	
Files to implement conjugate gradient in minFunc:
	applyNNetMinFunc.m   % Main file
	loss_func.m  % Loss function and its gradient



% The following files are supplied along with David Stutz's paper

Main file: 
	applyStochasticSquaredErrorTwoLayerPerceptronMNIST.m
	
Files to load and save the datasets:
	loadMNISTImages.m
	loadMNISTLabels.m
	saveMNISTImages.m

Sigmoid and its gradient:
	logisticSigmoid.m
	dLogisticSigmoid.m

Default SGD implementation:
	trainStochasticSquaredErrorTwoLayerPerceptron.m

Validation of the results:
	validateTwoLayerPerceptron.m