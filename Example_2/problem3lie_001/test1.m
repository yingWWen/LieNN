function [y_test] = test1(w,b,v,q)
%y_1 in the test set
    mm=size(q,2);
    y_test=zeros(1,mm);
    for i=1:mm
        y_test(i)=(1./(3.*sqrt(1111))).*(100.*exp(-0.01.*q(i))).*sin(((3.*sqrt(1111))./100).*q(i))+q(i).*sum(v.*sigmoid(q(i).*w+b));
    end
end
