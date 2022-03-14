function g = sigmoid(z)
%Activation functoon
g = 1.0 ./ (1.0 + exp(-z));
end
