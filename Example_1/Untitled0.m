x=-1:0.1:1;    %train set
q=-1.5:0.1:1.5;%test set
m=size(x,2);   %Number of training
mm=size(q,2);  %Number of testing
n=3;           %Number of neurons


w=rand_Init_Weights(n,1);   %N_1 Weights
b=rand_Init_Weights(n,1);   %N_1 Bias
v=rand_Init_Weights(n,1);   %N_1 Weights 
a=rand_Init_Weights(n,1);   %N_2 Weights
s=rand_Init_Weights(n,1);   %N_2 Bias
u=rand_Init_Weights(n,1);   %N_2 Weights

init_param=[w;b;v;a;s;u]; 

options=optimset('Display','off','GradObj', 'on','MaxIter', 10000); 
% Optimization during training and testing

%options=optimset('Display','iter','PlotFcn',{@optimplotx,@optimplotfval,@optimplotfirstorderopt},'GradObj', 'on','MaxIter', 10000);
%Iteration error during training

%options=optimset('Display','off','GradObj', 'on','MaxIter', 10000);
[param,cost,exit_flag]=...
    fminunc(@(p)(nnCostFunction(p,x,n)) , init_param, options);
% Optimization
disp(cost);   %Show loss function

w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);
a=param(3*n+1:4*n,:);
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);

y=predict1(w,b,v,x);    %y_1 in the training set
yy=predict2(a,s,u,x);   %y_2 in the training set
y_test=test1(w,b,v,q);  %y_1 in the test set
yy_test=test2(a,s,u,q); %y_2 in the test set 
y_r=sin(x);             %Exact solution y_1 in the training set
yy_r=1+x.^2;            %Exact solution y_2 in the training set
y_test_r=sin(q);        %Exact solution y_1 in the test set
yy_test_r=1+q.^2;       %Exact solution y_2 in the test set
error1=abs(y-y_r);      % Deviation of the network solution y_1 and the exact solution y_1 in the training set
error2=abs(y_test-y_test_r); %Deviation of the network solution y_1 and the exact solution y_1 in the test set

error3=abs(yy-yy_r);  % Deviation of the network solution y_2 and the exact solution y_2in the training set
error4=abs(yy_test-yy_test_r); %Deviation of the network solution y_2 and the exact solution y_2 in the test set

figure(1)
subplot(1,2,1)
plot(x,y,'r-x');
hold on;
plot(x,y_r,'k-o');  
hold on;
plot(x,yy,'b-o');
hold on;
plot(x,yy_r,'y-x');
xlabel('x')
ylabel('solution')
legend('The solution of Lie group NN y_1 ','Exact solution of y_1 ','The solution of Lie group NN y_2 ','Exact solution of y_2')
title('(1)')

% Training set image

subplot(1,2,2)
plot(q,y_test,'r-x');
hold on;
plot(q,y_test_r,'k-o');
hold on;
plot(q,yy_test,'b-o');
hold on;
plot(q,yy_test_r,'y-x');
xlabel('x')
ylabel('solution')
legend('The solution of Lie group NN y_1 ','Exact solution of y_1 ','The solution of Lie group NN y_2 ','Exact solution of y_2');
title('(2)')
hold on;

% Test set image
figure(2)
subplot(1,2,1)    
plot(x,error1,'-');
hold on;
plot(x,error3,'--');
hold on;
xlabel('x')
ylabel('Deviations \Delta y ')
legend('\Delta y_1 ','\Delta y_2 ')
title('\Delta y ')

%Training set deviation

subplot(1,2,2)    
plot(q,error2,'-');
hold on;
plot(q,error4,'--');
hold on;
xlabel('x')
ylabel('Deviations \Delta y ')
legend('\Delta y_1 ','\Delta y_2 ')
title('\Delta y ')

%Test set deviation

