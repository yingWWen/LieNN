function [y_test] = test1(w,b,v,q)
%y_1 in the test set
    mm=size(q,2);
    y_test=zeros(1,mm);
    for i=1:mm
        y_test(i)=(1./(sqrt(3))).*(2.*exp(-0.5.*q(i))).*sin(((sqrt(3))./2).*q(i))+q(i).*sum(v.*sigmoid(q(i).*w+b));
    end
end
