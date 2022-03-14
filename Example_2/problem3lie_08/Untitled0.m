x=0:0.1:2;     %Training set
q=0:0.1:2.5;   %Test set
A=rand(1,21);  %Random points in the training set
B=sort(A);     %Ascending order
m=size(x,2);   %Number of training points
mm=size(q,2);  %Number of test points
mmm=size(B,2); %Number of random points in the training set
n=3; %Number of neurons

%Parameter initialization
w=rand_Init_Weights(n,1);
b=rand_Init_Weights(n,1);
v=rand_Init_Weights(n,1);
a=rand_Init_Weights(n,1);
s=rand_Init_Weights(n,1);
u=rand_Init_Weights(n,1);

init_param=[w;b;v;a;s;u];

options=optimset('Display','off','GradObj', 'on','MaxIter', 10000);
%Optimization
% options=optimset('Display','iter','PlotFcn',{@optimplotx,@optimplotfval,@optimplotfirstorderopt},'GradObj', 'on','MaxIter', 10000);
%Number of iterations and loss function
[param,cost,exit_flag]=...
    fminunc(@(p)(nnCostFunction(p,x,n)) , init_param, options);
%Optimization
disp(cost);   %Display the loss function
disp(cal_error(param)); %Loss function for random points in the training set 
disp(pre_error(param));      %Loss function for the test set


w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);
a=param(3*n+1:4*n,:);
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);

y=predict1(w,b,v,x);    %Network solution y_1 in the training set
yy=predict2(a,s,u,x);   %Network solution y_2 in the training set
y_test=test1(w,b,v,q);  %Network solution y_1 in the test set
yy_test=test2(a,s,u,q); %Network solution y_2 in the test set
y_rand=predict1(w,b,v,B);  %Network solution for random points in the training set y_1
yy_rand=predict2(a,s,u,B); %Network solution for random points in the training set y_2
y_r=exp(-(8./5).*x).*sin(x);   %Exact solution y_1 in the training set
yy_r=exp(-(8./5).*x).*cos(x)-(8./5).*exp(-(8./5).*x).*sin(x); %Exact solution y_2 in the training set
y_test_r=exp(-(8./5).*q).*sin(q);   %Exact solution y_1 in the test set
yy_test_r=exp(-(8./5).*q).*cos(q)-(8./5).*exp(-(8./5).*q).*sin(q);  %Exact solution y_2 in the test set

y_a_r=exp(-(8./5).*B).*sin(B);   %Exact solution y_1 for random points in the training set
yy_a_r=exp(-(8./5).*B).*cos(B)-(8./5).*exp(-(8./5).*B).*sin(B); %Exact solution y_2 for random points in the training set




error1=abs(y-y_r);  % Deviation of the network solution y_1 and the exact solution in the training set
error2=abs(y_test-y_test_r); %  Deviation of the network solution y_1 and the exact solution in the test set

error3=abs(yy-yy_r);  %Deviation of the network solution y_2 and the exact solution in the training set
error4=abs(yy_test-yy_test_r); % Deviation of the network solution y_2 and the exact solution in the test set

derivation1=(sum(abs(y-y_r)))./m;  % Average of the deviation of the network solution y_1 and the exact solution y_1 at each point in the training set
derivation2=(sum(abs(y_rand-y_a_r)))./mmm;  % Average of the deviation of the network solution y_1 and the exact solution y_1 at random points in the training set
derivation3=(sum(abs(y_test-y_test_r)))./mm;  % % Average of the deviation of the network solution y_1 and the exact solution y_1 at each point in the test set

disp(derivation1);
disp(derivation2); 
disp(derivation3); 



figure(1);
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
legend('The solution of y_1 in Lie NN ','Exact solution of y_1 ','The solution of y_2 in Lie NN','Exact solution of y_2')
title('(1)')
%Training set image y_1 y_2


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
gca=legend('The solution of y_1 in Lie NN ','Exact solution of y_1 ','The solution of y_2 in Lie NN','Exact solution of y_2');
set( gca, 'Position', [0.65 0.70 0.08 0.2]);
title('(2)')
hold on;
%Test set image y_1 y_2

figure(2)
subplot(1,2,1)    %ѵ����0-2�ϵ� y_1�뾫ȷ������    �Լ� y_2�뾫ȷ������
plot(x,error1,'-');
hold on;
plot(x,error3,'--');
hold on;
xlabel('x')
ylabel('Deviations \Delta y ')
legend('\Delta y_1 ','\Delta y_2 ')
title('\Delta y ')



subplot(1,2,2)     %���Լ�0-1.5�ϵ� y_1�뾫ȷ������    �Լ� y_2�뾫ȷ������
plot(q,error2,'-');
hold on;
plot(q,error4,'--');
hold on;
xlabel('x')
ylabel('Deviations \Delta y ')
legend('\Delta y_1 ','\Delta y_2 ')
title('\Delta y ')


