%ʹ��dropout���������
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
W1=2*rand(20,25)-1;
W2=2*rand(20,20)-1;
W3=2*rand(20,20)-1;
W4=2*rand(5,20)-1;%����ṹ
N=5;
for epoch=1:10000
    [W1,W2,W3,W4]=DeepDropout(W1,W2,W3,W4,X,D,N);
end
for k=1:N
    x=reshape(X(:,:,k),25,1);
    v1=W1*x;
    y1=Sigmoid(v1);
    v2=W2*y1;
    y2=Sigmoid(v2);
    v3=W3*y2;
    y3=Sigmoid(v3);
    v4=W4*y3;
    y4=Softmax(v4);
    disp(y4);
end
%������һ�ε�������
function [W1,W2,W3,W4]=DeepDropout(W1,W2,W3,W4,X,D,N)
alpha=0.01;
for k=1:N
    x=reshape(X(:,:,k),25,1);
    d=D(k,:)';
    v1=W1*x;
    y1=Sigmoid(v1);
    y1=y1.*Dropout(y1,0.2);%���ز㶪��20%�Ľڵ㡣
    v2=W2*y1;
    y2=Sigmoid(v2);
    y2=y2.*Dropout(y2,0.2);
    v3=W3*y2;
    y3=Sigmoid(v3);
    y3=y3.*Dropout(y3,0.2);
    v4=W4*y3;
    y4=Softmax(v4);%ǰ�����
    e4=d-y4;delta4=e4;
    e3=W4'*delta4;delta3=y3.*(1-y3).*e3;
    e2=W3'*delta3;delta2=y2.*(1-y2).*e2;
    e1=W2'*delta2;delta1=y1.*(1-y1).*e1;%���򴫲�����
    dw4=alpha*delta4*y3';W4=W4+dw4;
    dw3=alpha*delta3*y2';W3=W3+dw3;
    dw2=alpha*delta2*y1';W2=W2+dw2;
    dw1=alpha*delta1*x';W1=W1+dw1;%Ȩֵ����
end
end
function y=Sigmoid(v)
y=1./(1+exp(-v));
end
function y=Softmax(v)
ex=exp(v);
y=ex/sum(ex);
end
function ym=Dropout(y,ratio)%���ؾ���Ϊ���Ԫ��ռ��ratio������Ԫ�����Ϊ1/��1-ratio)��
[m,n]=size(y);
ym=zeros(m,n);
num=round(m*n*(1-ratio));%������Ԫ����
idx=randperm(m*n,num);%���ѡȡ����Ԫ�ص�����
ym(idx)=1/(1-ratio);%��ֵ
end