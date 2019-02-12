%���ö����ķ���ʵ��ͼƬ����ʶ��
%����ṹ��25���룬5�����һ��50����
%����Ϊѵ������
clear all;
rng(3);
X=zeros(5,5,5);%����5x5����5����
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
%������һ�ε�������
function [W1,W2]=Multiclass(W1,W2,X,D,N)
alpha=0.9;
for k=1:N
    x=reshape(X(:,:,k),25,1);
    d=D(k,:)';
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Softmax(v2);%�����softmax����
    e2=d-y2;
    delta2=e2;%ʹ�ý����غ����Ľ��
    e1=W2'*delta2;%���򴫲�
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