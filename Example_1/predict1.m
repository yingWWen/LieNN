function [y] = predict1(w,b,v,x)
%The network solution y_1 in the training set
    m=size(x,2);
    y=zeros(1,m);
    for i=1:m
        y(i)=((1/2).*x(i)+4.*x(i).*cos(x(i))+(-4).*sin(x(i))+x(i).^2.*sin(x(i))+(1/4).*sin(2.*x(i)))+x(i).*sum(v.*sigmoid(x(i).*w+b));
    end
end
