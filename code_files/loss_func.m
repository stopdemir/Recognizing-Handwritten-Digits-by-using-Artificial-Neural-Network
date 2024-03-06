function [f, df] = loss_func(Y, X, w, numberOfHiddenUnits)

    function [S] = sigmoid(Z)
        S = 1 ./ (1 + exp(-Z));
    end
    
	[n, m] = size(Y);
    [~, b] = size(X);
	
    W1 = reshape(w(1:numberOfHiddenUnits*b), b, numberOfHiddenUnits);
    W2 = reshape(w(numberOfHiddenUnits*b+1:numberOfHiddenUnits*(b+m)), numberOfHiddenUnits , m);
	W1 = W1 ./ b;
    W2 = W2 ./ size(W2, 1);
    X = X ./ n;
    
	E = Y - sigmoid(sigmoid(X*W1)*W2);
	f = 0.5*sum(sum(E.*E));
	f = f/n;
    
	A1 = sigmoid(X*W1);
	A2 = sigmoid(A1*W2);
    ones_2 = ones(size(A2));    
    
	G2 = -A1' * (E .* A2 .* (ones_2 - A2));
	
	ones_1 = ones(size(A1));
	G1 = -X' * ((E .* A2 .* (ones_2 - A2)) * W2' .* (A1 .* (ones_1 - A1)));	
    
    df = [G2(:); G1(:)];
    df = df/n;

end
