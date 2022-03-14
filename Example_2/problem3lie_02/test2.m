function [yy_test] =test2(a,s,u,q)
%y_2 in the test set
    mm=size(q,2);
    yy_test=zeros(1,mm);
    for i=1:mm
        yy_test(i)=(1./12).*exp(-0.2.*q(i)).*(12.*cos(((2.*sqrt(6))./5).*q(i))-sqrt(6).*sin(((2.*sqrt(6))./5).*q(i)))+q(i).*sum(u.*sigmoid(q(i).*a+s));
    end
end
