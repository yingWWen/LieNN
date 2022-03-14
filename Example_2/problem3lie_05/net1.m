function output = net1(param,B)  
%Network_1
N1 = 0;
n=3;
w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);


for i = 1:n
    N1 = N1 + (sigmoid(B*w(i)+ b(i))*v(i));
end

output =  N1;
end

