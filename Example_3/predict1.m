function [y] = predict1(w,b,v,x)
%The network solution y_1 in the training set
    m=size(x,2);
    y=zeros(1,m);
    for i=1:m
        y(i)=cos(x(i))+x(i).*sum(v.*sigmoid(x(i).*w+b));
    end
end
