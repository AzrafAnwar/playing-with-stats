X = csvread('X_train.csv');
Xt = X.';
Y = csvread('Y_train.csv');

w_ls = inv(Xt*X)*Xt*Y;
lambda = linspace(0,5000,5001);

wRR = [];

for l = lambda
    wRR = [wRR, inv(l*eye(7)+Xt*X)*(Xt*X)*w_ls];
end

wrr1 = wRR(1,:);
wrr2 = wRR(2,:);
wrr3 = wRR(3,:);
wrr4 = wRR(4,:);
wrr5 = wRR(5,:);
wrr6 = wRR(6,:);
wrr7 = wRR(7,:);

dfl = [];
for l = lambda
    dfl = [dfl, trace(X*inv(Xt*X+l*eye(7))*Xt)];
end

figure;
hold
a1= plot(dfl,wrr1); M1 = "d1";
a2 = plot(dfl,wrr2);M2 = "d2";
a3 = plot(dfl,wrr3); M3 = "d3";
a4 = plot(dfl,wrr4); M4 = "d4"; 
a5 = plot(dfl,wrr5); M5 = "d5";
a6 = plot(dfl,wrr6); M6 = "d6";
a7 = plot(dfl,wrr7); M7 = "d7";
legend([a1,a2, a3, a4, a5, a6, a7], [M1, M2, M3, M4, M5, M6, M7]);

X2 = csvread('X_test.csv');
Y = csvread('Y_test.csv');
rmse = [];

for i = lambda+1
    Yp = X2*wRR(:,i);
    error = Y - Yp;
    re = (mean(error.^2))^.5;
    rmse = [rmse,re];
end

figure;
plot(lambda,rmse)


X = csvread('X_train.csv');
X = X.^2;
Xt = X.';
Y = csvread('Y_train.csv');

w_ls = inv(Xt*X)*Xt*Y;
lambda = linspace(0,5000,5001);

wRR = [];

for l = lambda
    wRR = [wRR, inv(l*eye(7)+Xt*X)*(Xt*X)*w_ls];
end

wrr1 = wRR(1,:);
wrr2 = wRR(2,:);
wrr3 = wRR(3,:);
wrr4 = wRR(4,:);
wrr5 = wRR(5,:);
wrr6 = wRR(6,:);
wrr7 = wRR(7,:);

dfl = [];
for l = lambda
    dfl = [dfl, trace(X*inv(Xt*X+l*eye(7))*Xt)];
end

figure;
hold
a1= plot(dfl,wrr1); M1 = "d1";
a2 = plot(dfl,wrr2);M2 = "d2";
a3 = plot(dfl,wrr3); M3 = "d3";
a4 = plot(dfl,wrr4); M4 = "d4"; 
a5 = plot(dfl,wrr5); M5 = "d5";
a6 = plot(dfl,wrr6); M6 = "d6";
a7 = plot(dfl,wrr7); M7 = "d7";
legend([a1,a2, a3, a4, a5, a6, a7], [M1, M2, M3, M4, M5, M6, M7]);

X2 = csvread('X_test.csv');
Y = csvread('Y_test.csv');
rmse = [];

for i = lambda+1
    Yp = X2*wRR(:,i);
    error = Y - Yp;
    re = (mean(error.^2))^.5;
    rmse = [rmse,re];
end

figure;
plot(lambda,rmse)



X = csvread('X_train.csv');
X = X.^3;
Xt = X.';
Y = csvread('Y_train.csv');

w_ls = inv(Xt*X)*Xt*Y;
lambda = linspace(0,5000,5001);

wRR = [];

for l = lambda
    wRR = [wRR, inv(l*eye(7)+Xt*X)*(Xt*X)*w_ls];
end

wrr1 = wRR(1,:);
wrr2 = wRR(2,:);
wrr3 = wRR(3,:);
wrr4 = wRR(4,:);
wrr5 = wRR(5,:);
wrr6 = wRR(6,:);
wrr7 = wRR(7,:);

dfl = [];
for l = lambda
    dfl = [dfl, trace(X*inv(Xt*X+l*eye(7))*Xt)];
end

figure;
hold
a1= plot(dfl,wrr1); M1 = "d1";
a2 = plot(dfl,wrr2);M2 = "d2";
a3 = plot(dfl,wrr3); M3 = "d3";
a4 = plot(dfl,wrr4); M4 = "d4"; 
a5 = plot(dfl,wrr5); M5 = "d5";
a6 = plot(dfl,wrr6); M6 = "d6";
a7 = plot(dfl,wrr7); M7 = "d7";
legend([a1,a2, a3, a4, a5, a6, a7], [M1, M2, M3, M4, M5, M6, M7]);

X2 = csvread('X_test.csv');
Y = csvread('Y_test.csv');
rmse = [];

for i = lambda+1
    Yp = X2*wRR(:,i);
    error = Y - Yp;
    re = (mean(error.^2))^.5;
    rmse = [rmse,re];
end

figure;
plot(lambda,rmse)