x=0:0.10:2;    % Training set 
q=0:0.10:2.5;  % Test set 
m=size(x);     %Number of training sets
mm=size(q);    %Number of test sets
n=3;           %Number of neurons

w=rand_Init_Weights(n,1);   %Initialization of weights y_1
b=rand_Init_Weights(n,1);   %Initialization of bias y_1
v=rand_Init_Weights(n,1);   %Initialization of weights y_1
a=rand_Init_Weights(n,1);   %Initialization of weights y_2
s=rand_Init_Weights(n,1);   %Initialization of bias y_2
u=rand_Init_Weights(n,1);   %Initialization of weights y_2

init_param=[w;b;v;a;s;u]; 

options=optimset('Display','off','GradObj', 'on','MaxIter', 10000);
% Optimization during training and testing

%options=optimset('Display','iter','PlotFcn',{@optimplotx,@optimplotfval,@optimplotfirstorderopt},'GradObj', 'on','MaxIter', 10000);
%Iteration error during training

[param,cost,exit_flag]=...
    fminunc(@(p)(nnCostFunction(p,x,n)) , init_param, options);
%Optimization
disp(cost);  % Show loss function

w=param(1:n,:);
b=param(n+1:2*n,:);
v=param(2*n+1:3*n,:);
a=param(3*n+1:4*n,:);
s=param(4*n+1:5*n,:);
u=param(5*n+1:6*n,:);



y=predict1(w,b,v,x);   %y_1 in the training set
yy=predict2(a,s,u,x);  %y_2 in the training set
y_test=test1(w,b,v,q);  %y_1 in the test set
yy_test=test2(a,s,u,q);  %y_2 in the test set

dy = @(t, y_ode)[y_ode(2); (-y_ode(1)-y_ode(1)^3)];
y10 = 1;
y20 = 0;
tspan = [0:0.1:2];

[t, y_ode] = ode45(dy, tspan, [y10, y20]);
% RK method in the training set y_1 y_2


dyy = @(tt, y_ode_test)[y_ode_test(2); (-y_ode_test(1)-y_ode_test(1)^3)];
y10 = 1;
y20 = 0;
tspan = [0:0.1:2.5];

[tt, y_ode_test] = ode45(dyy, tspan, [y10, y20]);
% RK method in the test set y_1 y_2


figure(1)
plot(x,y,'r-x');
hold on;
plot(t,y_ode(:, 1),'b-o');
hold on;
plot(x,yy,'r-x');
hold on;
plot(t,y_ode(:, 2),'b-o');
hold on;
xlabel('x')
ylabel('solution')
legend('The solution of y_1 in Lie NN ','RK solution of y_1','The solution of y_2 in Lie NN ','RK solution of y_2')
%title('(1)')
%Training set image

figure(2)
plot(q,y_test,'r-x');
hold on;
plot(tt,y_ode_test(:, 1),'b-o');
hold on;
plot(q,yy_test,'r-x');
hold on;
plot(tt,y_ode_test(:, 2),'b-o');
hold on;
xlabel('x')
ylabel('solution')
legend('The solution of y_1 in Lie NN ','RK solution of y_1','The solution of y_2 in Lie NN ','RK solution of y_2')
%title('(2)')
%Test set image























% 
% subplot(1,2,2)
% plot(q,y_test,'r-x');
% hold on;
% plot(q,RK_ytest,'k-o');
% hold on;
% plot(q,yy_test,'b-o');
% hold on;
% plot(q,RK_ztest,'y-x');
% xlabel('x')
% ylabel('solution')
% gca=legend('The solution of y_1 in Lie NN ','RK solution of y_1 ','The solution of y_2 in Lie NN','RK solution of y_2');
% set( gca, 'Position', [0.65 0.70 0.08 0.2]);
% title('(2)')


% figure(1)
% plot(x,y,'r-x');
% hold on;
% plot(x,yy,'r-x');%lie神经网络方法
% hold on;
% plot(t,xx,'k-d'); %这样是对的，因为不能写成plot(t,y(1),'r-x',t,y(2),'g-x')),如果写成这样，就会直接画成两个初始条件的图形！这是ode45方法，也就是龙格库塔方法的四阶五阶算法.
% hold on; 
% plot(x,RK_y,'b-+');
% hold on;
% plot(x,RK_z,'b-+');%龙格库塔方法
% hold on;
% 
% 

% figure(2)
% plot(x,y,'r-x');
% hold on;
% plot(x,yy,'r-x');%lie神经网络方法
% hold on;
% plot(t,xx,'k-d'); %这样是对的，因为不能写成plot(t,y(1),'r-x',t,y(2),'g-x')),如果写成这样，就会直接画成两个初始条件的图形！这是ode45方法，也就是龙格库塔方法的四阶五阶算法.
% hold on; 
% plot(x,RK_y,'b-+');
% hold on;
% plot(x,RK_z,'b-+');%龙格库塔方法
% hold on;
% 
% 
% figure(3)
% subplot(1,2,1)
% plot(x,y,'r-x');
% hold on;
% plot(x,RK_y,'k-o');
% hold on;
% plot(x,yy,'b-o');
% hold on;
% plot(x,RK_z,'y-x');
% xlabel('x')
% ylabel('solution')
% legend('The solution of y_1 in Lie NN ','RK solution of y_1 ','The solution of y_2 in Lie NN','RK solution of y_2')
% title('(1)')
% 
% 
% 
% subplot(1,2,2)
% plot(q,y_test,'r-x');
% hold on;
% plot(q,RK_ytest,'k-o');
% hold on;
% plot(q,yy_test,'b-o');
% hold on;
% plot(q,RK_ztest,'y-x');
% xlabel('x')
% ylabel('solution')
% gca=legend('The solution of y_1 in Lie NN ','RK solution of y_1 ','The solution of y_2 in Lie NN','RK solution of y_2');
% title('(2)')
% 
% 
% 
% % plot(x,y_euler,'g-o');
% % hold on;
% % plot(x,z_euler,'g-o');%欧拉方法
% % hold on;
% 
% 
% 
% figure(2)
% plot(x,y,'r-x');
% hold on;
% plot(x,yy,'r-x');%lie神经网络方法
% hold on;
% plot(t,xx,'k-d'); %这样是对的，因为不能写成plot(t,y(1),'r-x',t,y(2),'g-x')),如果写成这样，就会直接画成两个初始条件的图形！这是ode45方法，也就是龙格库塔方法的四阶五阶算法.
% hold on; 
% plot(x,RK_y,'b-+');
% hold on;
% plot(x,RK_z,'b-+');%龙格库塔方法
% hold on;
% 
% 
% 
% figure(3)
% plot(q,y_test,'r-x');
% hold on;
% plot(q,yy_test,'r-x');%lie神经网络方法
% hold on;
% plot(q,RK_ytest,'b-+');
% hold on;
% plot(q,RK_ztest,'b-+');%龙格库塔方法
% hold on;
% % plot(tt,z,'b-s');  %ode45
% % hold on;
% 
