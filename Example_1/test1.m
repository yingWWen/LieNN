function [y_test] = test1(w,b,v,q)
%The network solution y_1 in the testing set
    mm=size(q,2);
    y_test=zeros(1,mm);
    for i=1:mm
        y_test(i)=((1/2).*q(i)+4.*q(i).*cos(q(i))+(-4).*sin(q(i))+q(i).^2.*sin(q(i))+(1/4).*sin(2.*q(i)))+q(i).*sum(v.*sigmoid(q(i).*w+b));
    end
end

