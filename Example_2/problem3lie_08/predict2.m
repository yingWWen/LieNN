function [yy] = predict2(a,s,u,x)
%y_2 in the training set
    m=size(x,2);
    yy=zeros(1,m);
    for i=1:m
        yy(i)=((1/3).*exp(1).^((-4/5).*x(i)).*(3.*cos((3/5).*x(i))+(-4).*sin((3/5).*x(i))))+x(i).*sum(u.*sigmoid(x(i).*a+s));
    end
end
