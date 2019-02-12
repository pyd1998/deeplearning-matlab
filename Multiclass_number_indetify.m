%运用多分类的方法实现图片数字识别
%网络结构：25输入，5输出，一层50隐藏
%以下为训练代码
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
W1=2*rand(50,25)-1;
W2=2*rand(5,50)-1;
N=5;
for epoch=1:10000
    [W1,W2]=Multiclass(W1,W2,X,D,N);
end
for k=1:N
    x=reshape(X(:,:,k),25,1);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Softmax(v2);
    disp(y2);
end
%以下是一次迭代过程
function [W1,W2]=Multiclass(W1,W2,X,D,N)
alpha=0.9;
for k=1:N
    x=reshape(X(:,:,k),25,1);
    d=D(k,:)';
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Softmax(v2);%输出层softmax激活
    e2=d-y2;
    delta2=e2;%使用交叉熵函数的结果
    e1=W2'*delta2;%反向传播
    delta1=y1.*(1-y1).*e1;
    dw1=alpha*delta1*x';
    dw2=alpha*delta2*y1';
    W1=W1+dw1;
    W2=W2+dw2;
end
end
function y=Sigmoid(v)
y=1./(1+exp(-v));
end
function y=Softmax(v)
ex=exp(v);
y=ex/sum(ex);
end