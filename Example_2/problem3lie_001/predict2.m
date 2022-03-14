function [yy] = predict2(a,s,u,x)
%y_2 in the training set
    m=size(x,2);
    yy=zeros(1,m);
    for i=1:m
        yy(i)=(1./3333).*exp(-0.01.*x(i)).*(3333.*cos(((3.*sqrt(1111))./100).*x(i))-sqrt(1111).*sin(((3.*sqrt(1111))./100).*x(i)))+x(i).*sum(u.*sigmoid(x(i).*a+s));
    end
end
