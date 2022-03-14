function [E,grad] = nnCostFunction(param,x,n)
% E:Loss function  grad:Optimized parameters
w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);
a=param(3*n+1:4*n,:);
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);


m=size(x,2);
n=size(w,1);
J=0;
JJ=0;
grad_w=zeros(n,1);
grad_b=zeros(n,1);
grad_v=zeros(n,1);
grad_a=zeros(n,1);
grad_s=zeros(n,1);
grad_u=zeros(n,1);

for i=1:m
    sum=0;
    som=0;
    num=0;
    nom=0;
    for j=1:n
        sigma=sigmoid(w(j).*x(i)+b(j));
        sum=sum+v(j).*sigma;
        som=som+v(j).*sigma.*(1+w(j).*x(i).*(1-sigma));
        sigm=sigmoid(a(j).*x(i)+s(j));
        num=num+u(j).*sigm;
        nom=nom+u(j).*sigm.*(1+(1./50).*x(i)+a(j).*x(i).*(1-sigm));
    end
    tmp=(som-x(i).*num);
    J=J+(0.5/m).*tmp^2;
    tmp_sigma=sigmoid(x(i).*w+b);
    %obtained from the equation
    tmpe=((1./50).*exp(-(0.02.*x(i))).*cos(x(i))+nom+x(i).*sum);
    JJ=JJ+(0.5/m).*tmpe^2;
    tmpe_sigm=sigmoid(x(i).*a+s);
    %obtained from the equation 
    E=J+JJ;   
    %Loss function
    grad_w=grad_w + (1/m).*tmp.*(...
        2.*x(i).*(v.*tmp_sigma.*(1-tmp_sigma)) +...
        x(i).*x(i)*(v.*w.*tmp_sigma.*(1-tmp_sigma).*(1-2*tmp_sigma)))+...
        (1/m).*tmpe.*(x(i).*x(i).*(v.*tmp_sigma.*(1-tmp_sigma)));
    %Derivative of E with respect to the weights y_1
    grad_b=grad_b + (1/m).*tmp.*(...
        tmp_sigma.*(1-tmp_sigma).*v+...
        x(i).*tmp_sigma.*(1-tmp_sigma).*(1-2*tmp_sigma).*v.*w)+...
        (1/m).*tmpe.*(x(i).*(v.*tmp_sigma.*(1-tmp_sigma)));
    %Derivative of E with respect to the bias y_1
    grad_v=grad_v + (1/m).*tmp.*(...
        tmp_sigma + x(i).*w.*tmp_sigma.*(1-tmp_sigma))+...
        (1/m).*tmpe.*(x(i).*tmp_sigma);
    %Derivative of E with respect to the weights y_1
    grad_a=grad_a + (1/m).*tmp.*(...
        -x(i).*x(i).*u.*tmpe_sigm.*(1-tmpe_sigm))+...
        (1/m).*tmpe.*(2.*x(i).*(u.*tmpe_sigm.*(1-tmpe_sigm))+(1./50).*x(i).*x(i).*(u.*tmpe_sigm.*(1-tmpe_sigm))+...
        x(i).*x(i).*(a.*u.*tmpe_sigm.*(1-tmpe_sigm).*(1-2.*tmpe_sigm)));
    %Derivative of E with respect to the weights y_2
    grad_s=grad_s + (1/m).*tmp.*(...
        -x(i).*u.*tmpe_sigm.*(1-tmpe_sigm))+...
        (1/m).*tmpe.*((u.*tmpe_sigm.*(1-tmpe_sigm))+(1./50).*x(i).*(u.*tmpe_sigm.*(1-tmpe_sigm))+...
        x(i).*(a.*u.*tmpe_sigm.*(1-tmpe_sigm).*(1-2*tmpe_sigm)));
   %%Derivative of E with respect to the bias y_2
    
    grad_u=grad_u + (1/m).*tmp.*(...
      -x(i).*tmpe_sigm)+...
       (1/m).*tmpe.*(tmpe_sigm+(1./50).*x(i).*tmpe_sigm+x(i).*a.*tmpe_sigm.*(1-tmpe_sigm));
  %Derivative of E with respect to the weights y_2
end

grad=[grad_w;grad_b;grad_v;grad_a;grad_s;grad_u];
%grad:Updated parameters
end
