function [yy_test] =test2(a,s,u,q)
%y_2 in the test set
    mm=size(q,2);
    yy_test=zeros(1,mm);
    for i=1:mm
        yy_test(i)=((1/3).*exp(1).^((-4/5).*q(i)).*(3.*cos((3/5).*q(i))+(-4).*sin((3/5).*q(i))))+q(i).*sum(u.*sigmoid(q(i).*a+s));
    end
end
