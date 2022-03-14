function [y] = predict1(w,b,v,x)
%y_1 in the training set
    m=size(x,2);
    y=zeros(1,m);
    for i=1:m
        y(i)=((5/3).*exp(1).^((-4/5).*x(i)).*sin((3/5).*x(i)))+x(i).*sum(v.*sigmoid(x(i).*w+b));
    end
end
