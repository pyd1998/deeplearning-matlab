%���������磬����ݶ��㷨�������㷨�ıȽ�
%����Ϊ���Դ���
clear all;
X=[0 0 1;
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 0 1 1];
E1=zeros(1000,1);
E2=zeros(1000,1);
W1=2*rand(1,3)-1;
W2=W1;
N=4;
for epoch=1:1000
    W1=DeltaSGD(W1,X,D,N);
    W2=DeltaBatch(W2,X,D,N);
    es1=0;
    es2=0;
    for k=1:N    %�������ƽ��ֵ
        x=X(k,:)';
        d=D(k);%��ǩ׼ȷֵ
        v1=W1*x;
        y1=Sigmoid(v1);%����ݶȷ��Ľ��
        es1=es1+(d-y1)^2;%����ۼ�
        v2=W2*x;
        y2=Sigmoid(v2);%�����㷨�Ľ��
        es2=es2+(d-y2)^2;%����ۼ�
    end
    E1(epoch)=es1/N;%����ÿһ�ֺ�����ƽ��ֵ
    E2(epoch)=es2/N;
end
plot(E1,'r');
hold on
plot(E2,'b');
xlabel('Epoch');
ylabel('ѵ�����ƽ��ֵ');
legend('SGD','Batch');

function W=DeltaSGD(W,X,D,N)
%WΪȨ�ؾ���
%XΪ�������
%DΪ��ȷ���
alpha=0.9;          %learning rate                
for k=1:N
    x=X(k,:)';         %ѡȡһ������
    d=D(k);         %��Ӧ�ı�ǩֵ
    v=W*x;          %ǰ�򴫲�
    y=Sigmoid(v);   %����
    e=d-y;          %���ֵ
    delta=y*(1-y)*e;
    dw=alpha*delta*x;%����������ʽ
    W(1)=W(1)+dw(1);
    W(2)=W(2)+dw(2);
    W(3)=W(3)+dw(3);%Ȩֵ����
end
end

function W=DeltaBatch(W,X,D,N)
alpha=0.9;
dWsum=zeros(3,1);
for k=1:N
    x=X(k,:)';
    d=D(k);
    v=W*x;
    y=Sigmoid(v);
    e=d-y;
    delta=y*(1-y)*e;
    dw=alpha*delta*x;
    dWsum=dWsum+dw;
end
dWavg=dWsum/N; %ÿ��ֻ��һ�ε���
W(1)=W(1)+dWavg(1);
W(2)=W(2)+dWavg(2);
W(3)=W(3)+dWavg(3);
end

function y=Sigmoid(v)
y=1/(1+exp(-v));
end