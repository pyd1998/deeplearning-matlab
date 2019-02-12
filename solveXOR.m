%���÷��򴫲��㷨���XOR����
%����ṹ�������룬һ�����һ���Ľڵ�����,����SGD����
%����Ϊѵ������
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
    [W1,W2]=BackpropXOR(W1,W2,X,D,N);
end
for k=1:N    %���ղ���
    x=X(k,:)';
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);
    disp(y2);
end
%����Ϊһ�ε����Ĺ���
function [W1,W2]=BackpropXOR(W1,W2,X,D,N)
alpha=0.9;
for k=1:N
    x=X(k,:)';
    d=D(k);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);%ǰ�����
    e=d-y2;
    delta2=y2.*(1-y2).*e;
    e1=W2'*delta2;%���򴫲�
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