function [yy] = predict2(a,s,u,x)
%The network solution y_2 in the training set
    m=size(x,2);
    yy=zeros(1,m);
    for i=1:m
        yy(i)=-sin(x(i))+x(i).*sum(u.*sigmoid(x(i).*a+s));
    end
end
