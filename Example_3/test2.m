function [yy_test] =test2(a,s,u,q)
%The network solution y_2 in the test set
    mm=size(q,2);
    yy_test=zeros(1,mm);
    for i=1:mm
        yy_test(i)=-sin(q(i))+q(i).*sum(u.*sigmoid(q(i).*a+s));
    end
end
