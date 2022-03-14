function JVal2 = cost2(param,B) 
%Left side of the equation minus the right side
N1 = net1(param,B); %Network_1
N2 = net2(param,B); %Network_2
n=3;
dN2_dx = 0;
a=param(3*n+1:4*n,:);   
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);
for i = 1:n
    dN2_dx = dN2_dx + u(i)*a(i)*dsigmoid(B*a(i)+s(i));    %Derivative of network_2
end


JVal2 =(exp(-B).*cos(B)+(1+B)*N2+B*dN2_dx+B*N1);  %Left side of the equation minus the right side

end

