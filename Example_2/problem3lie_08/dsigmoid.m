function y = dsigmoid(x)
%The derivative of the activation function

n = sigmoid(x);
y = n*(1-n);

end


