function [y_test] = test1(w,b,v,q)
%The network solution y_1 in the test set
    mm=size(q,2);
    y_test=zeros(1,mm);
    for i=1:mm
        y_test(i)=cos(q(i))+q(i).*sum(v.*sigmoid(q(i).*w+b));
    end
end
