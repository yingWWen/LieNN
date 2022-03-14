function jVal = cal_error(param)  
%%Loss function for random points in the training set 


A=rand(1,21);
B=sort(A);     %%Random points in the training set
mmm=size(B,2);


jVal1 = 0;
jVal2 = 0;


for i = 1:mmm      
    jVal1 = jVal1+(1/(2*mmm))*(cost1(param,B(i)))^2 ;
    jVal2 = jVal2+(1/(2*mmm))*(cost2(param,B(i)))^2;
end

jVal=jVal1+jVal2;
end

