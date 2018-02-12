clear all
close all

load cbfxmpl.mat

%FOR AN MA FILTER WITH Q = 30

x = cbf;
y = cbff2;

X = toeplitz(x,[x(1) zeros(1,30)]);
Xt = X.';
b = inv(Xt*X)*Xt*y;


figure(1)
plot(y)
title("Output Signal from Dataset,cbff2")

figure(2)
plot(x)
title("Input Signal from Dataset, cbf")

figure(3)
plot(X*b)
title("Predicted Output Signal with generated MA filter")

figure(4)
plot(b)
title("MA Filter response, b")

%FOR ARMA FILTER WITH P = 5 and Q = 45 

Y = toeplitz(y,[y(1) zeros(1,5)]);
X2 = toeplitz(x,[x(1), zeros(1,45)]);

Y_X = [Y, X2];

sol = (inv((Y_X.')*(Y_X))*(Y_X.')*y);
a = -sol(1:6);
b = sol(7:52);

figure
plot(a)
title("ARMA Filter response, a")

figure
plot(b)
title("ARMA Filter response, b")

figure
plot(y)
title("Output Signal from Dataset,cbff2")

figure
plot(x)
title("Input Signal from Dataset, cbf")

figure
plot((Y*a+X2*b)/3)
title("Predicted Output Signal with generated ARMA filter")



