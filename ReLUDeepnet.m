%使用relu代替Sigmoid进行深度网络构建
%以下为训练代码。
clear all;
rng(3);
X=zeros(5,5,5);%输入5x5矩阵5个。
X(:,:,1)=[0 1 1 0 0;
    0 0 1 0 0;
    0 0 1 0 0;
    0 0 1 0 0;
    0 1 1 1 0];
X(:,:,2)=[1 1 1 1 0;
    0 0 0 0 1;
    0 1 1 1 0;
    1 0 0 0 0;
    1 1 1 1 1];
X(:,:,3)=[1 1 1 1 0;
    0 0 0 0 1;
    0 1 1 1 0;
    0 0 0 0 1;
    1 1 1 1 0];
X(:,:,4)=[0 0 0 1 0;
    0 0 1 1 0;
    0 1 0 1 0;
    1 1 1 1 1;
    0 0 0 1 0];
X(:,:,5)=[1 1 1 1 1;
    1 0 0 0 0;
    1 1 1 1 0;
    0 0 0 0 1;
    1 1 1 1 0];
D=[1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1];
W1=2*rand(20,25)-1;
W2=2*rand(20,20)-1;
W3=2*rand(20,20)-1;
W4=2*rand(5,20)-1;%网络结构
N=5;
for epoch=1:10000
    [W1,W2,W3,W4]=DeepReLU(W1,W2,W3,W4,X,D,N);
end
for k=1:N
    x=reshape(X(:,:,k),25,1);
    v1=W1*x;
    y1=ReLU(v1);
    v2=W2*y1;
    y2=ReLU(v2);
    v3=W3*y2;
    y3=ReLU(v3);
    v4=W4*y3;
    y4=Softmax(v4);
    disp(y4);
end
%以下为一次迭代过程
function [W1,W2,W3,W4]=DeepReLU(W1,W2,W3,W4,X,D,N)
alpha=0.01;%学习率变小了
for k=1:N
    x=reshape(X(:,:,k),25,1);
    d=D(k,:)';
    v1=W1*x;
    y1=ReLU(v1);
    v2=W2*y1;
    y2=ReLU(v2);
    v3=W3*y2;
    y3=ReLU(v3);
    v4=W4*y3;
    y4=Softmax(v4);%前向过程
    e4=d-y4;delta4=e4;
    e3=W4'*delta4;delta3=(v3>0).*e3;
    e2=W3'*delta3;delta2=(v2>0).*e2;
    e1=W2'*delta2;delta1=(v1>0).*e1;%反向传播过程
    dw4=alpha*delta4*y3';
    dw3=alpha*delta3*y2';
    dw2=alpha*delta2*y1';
    dw1=alpha*delta1*x';%计算增量
    W4=W4+dw4;W3=W3+dw3;W2=W2+dw2;W1=W1+dw1;%权值更新
end
end
function y=ReLU(v)
y=max(0,v);
end
function y=Softmax(v)
ex=exp(v);
y=ex/sum(ex);
end