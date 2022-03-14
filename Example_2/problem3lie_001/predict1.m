function [y] = predict1(w,b,v,x)
%y_1 in the training set
    m=size(x,2);
    y=zeros(1,m);
    for i=1:m
        y(i)=(1./(3.*sqrt(1111))).*(100.*exp(-0.01.*x(i))).*sin(((3.*sqrt(1111))./100).*x(i))+x(i).*sum(v.*sigmoid(x(i).*w+b));
    end
end
