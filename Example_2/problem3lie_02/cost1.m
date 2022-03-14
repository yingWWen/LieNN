function JVal1 = cost1(param,B)  
%Left side of the equation minus the right side
N1 = net1(param,B);  %Network_1
N2 = net2(param,B);  %Network_2
n=3;
w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);

dN1_dx = 0;
for i = 1:n       
    dN1_dx = dN1_dx + v(i)*w(i)*dsigmoid(B*w(i)+b(i));  %Derivative of network_1
end

d_fai1 = N1 + B* dN1_dx;  

JVal1 =  d_fai1 -B*N2;   %Left side of the equation minus the right side

end

