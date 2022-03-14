function [yy] = predict2(a,s,u,x)
%y_2 in the training set
    m=size(x,2);
    yy=zeros(1,m);
    for i=1:m
        yy(i)=(1./12).*exp(-0.2.*x(i)).*(12.*cos(((2.*sqrt(6))./5).*x(i))-sqrt(6).*sin(((2.*sqrt(6))./5).*x(i)))+x(i).*sum(u.*sigmoid(x(i).*a+s));
    end
end
