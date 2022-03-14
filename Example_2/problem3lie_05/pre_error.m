function jVal = pre_error(param)   
%%Loss function for the test set

q=0:0.1:2.5;%%Test set
mm=size(q,2);%%Number of test points



jVal1 = 0;
jVal2 = 0;


for i = 1:mm       
    jVal1 = jVal1+(1/(2*mm))*(cost1(param,q(i)))^2 ;
    jVal2 = jVal2+(1/(2*mm))*(cost2(param,q(i)))^2;
end

jVal=jVal1+jVal2;
end

