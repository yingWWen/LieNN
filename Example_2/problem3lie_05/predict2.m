function [yy] = predict2(a,s,u,x)
%y_2 in the training set
    m=size(x,2);
    yy=zeros(1,m);
    for i=1:m
        yy(i)=(1./3).*exp(-0.5.*x(i)).*(3.*cos(((sqrt(3))./2).*x(i))-sqrt(3).*sin(((sqrt(3))./2).*x(i)))+x(i).*sum(u.*sigmoid(x(i).*a+s));
    end
end
