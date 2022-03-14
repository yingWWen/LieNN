function y = dsigmoid(x)
%Derivative of the activation function  dsigmoid(x)=sigmoid(x)(1-sigmoid(x))

n = sigmoid(x);
y = n*(1-n);

end


