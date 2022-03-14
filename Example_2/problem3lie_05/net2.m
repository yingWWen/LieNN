function output = net2(param,B)   
%%Network_2
N2 = 0;
n=3;
a=param(3*n+1:4*n,:);   
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);


for i = 1:n
    N2 = N2 + (sigmoid(B*a(i)+ s(i))*u(i));
end

output =  N2;
end

