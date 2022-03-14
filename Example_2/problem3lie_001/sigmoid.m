function g = sigmoid(z)
%Activation function
g = 1.0 ./ (1.0 + exp(-z));
end
