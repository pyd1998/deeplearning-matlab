clear all;
X=[0 0 1;
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 1 1 0];
E1=zeros(1000,1);
E2=zeros(1000,1);
W11=2*rand(4,3)-1;
W12=2*rand(1,4)-1;
W21=W11;
W22=W12;
N=4;
for epoch=1:1000
    [W11,W12]=BackpropCE(W11,W12,X,D,N);
    [W21,W22]=BackpropXOR(W21,W22,X,D,N);
    es1=0;es2=0;
    for k=1:N
        x=X(k,:)';
        d=D(k);
        es1=es1+(d-Sigmoid(W12*Sigmoid(W11*x)))^2;
        es2=es2+(d-Sigmoid(W22*Sigmoid(W21*x)))^2;
    end
    E1(epoch)=es1/N;
    E2(epoch)=es2/N;
end
plot(E1,'r');
hold on;
plot(E2,'b:');
xlabel('epoch');
ylabel('average of training error');
legend('交叉熵','误差平方和');

function [W1,W2]=BackpropCE(W1,W2,X,D,N)
alpha=0.9;
for k=1:N
    x=X(k,:)';
    d=D(k);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);%前向过程
    e2=d-y2;
    delta2=e2;%使用交叉熵函数的delta2计算方法
    e1=W2'*delta2;%反向传播
    delta1=y1.*(1-y1).*e1;
    dw1=alpha*delta1*x';
    dw2=alpha*delta2*y1';
    W1=W1+dw1;
    W2=W2+dw2;
end
end

function [W1,W2]=BackpropXOR(W1,W2,X,D,N)
alpha=0.9;
for k=1:N
    x=X(k,:)';
    d=D(k);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);%前向过程
    e=d-y2;
    delta2=y2.*(1-y2).*e;
    e1=W2'*delta2;%反向传播
    delta1=y1.*(1-y1).*e1;
    dw1=alpha*delta1*x';
    W1=W1+dw1;
    dw2=alpha*delta2*y1';
    W2=W2+dw2;
end
end
function y=Sigmoid(v)
y=1./(1+exp(-v));
end