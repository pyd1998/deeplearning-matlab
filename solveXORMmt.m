%运用动量法解决XOR问题
%以下为训练代码
clear all;
X=[0 0 1;
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 1 1 0];
W1=2*rand(4,3)-1;
W2=2*rand(1,4)-1;
N=4;%dataset size
for epoch=1:10000
    [W1,W2]=BackpropMmt(W1,W2,X,D,N);
end
for k=1:N    %最终测试
    x=X(k,:)';
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);
    disp(y2);
end
%以下为一次迭代过程
function [W1,W2]=BackpropMmt(W1,W2,X,D,N)
alpha=0.9;
beta=0.9;
mmt1=zeros(size(W1));
mmt2=zeros(size(W2));
for k=1:N
    x=X(k,:)';
    d=D(k);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);%前向过程
    e2=d-y2;
    delta2=y2.*(1-y2).*e2;
    e1=W2'*delta2;%反向传播
    delta1=y1.*(1-y1).*e1;
    dw1=alpha*delta1*x';
    dw2=alpha*delta2*y1';
    mmt1=dw1+beta*mmt1;%动量叠加
    W1=W1+mmt1;
    mmt2=dw2+beta*mmt2;%动量叠加
    W2=W2+mmt2;
end
end
function y=Sigmoid(v)
y=1./(1+exp(-v));
end