function [yy_test] = test2(a,s,u,q)
%The network solution y2 in the testing set
    mm=size(q,2);
    yy_test=zeros(1,mm);
    for i=1:mm
        yy_test(i)=(2+q(i).^2+(-1).*cos(q(i))+q(i).^2.*cos(q(i))+(-2).*q(i).*sin(q(i)))+q(i).*sum(u.*sigmoid(q(i).*a+s));
    end
end

