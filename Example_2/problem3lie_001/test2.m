function [yy_test] =test2(a,s,u,q)
% y_2 in the test set
    mm=size(q,2);
    yy_test=zeros(1,mm);
    for i=1:mm
        yy_test(i)=(1./3333).*exp(-0.01.*q(i)).*(3333.*cos(((3.*sqrt(1111))./100).*q(i))-sqrt(1111).*sin(((3.*sqrt(1111))./100).*q(i)))+q(i).*sum(u.*sigmoid(q(i).*a+s));
    end
end
