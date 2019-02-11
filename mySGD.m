%���������磬����ݶ��½�
%����Ϊ���Դ���
clear all;
N=4;   %dataset size
X=[0 0 1;   %input
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 0 1 1]; %right output
W=2*rand(1,3)-1; %start weights
for epoch=1:100000      %����100000�ֵ���
    W=DeltaSGD(W,X,D,N);
end

for k=1:N         %���ղ���
    x=X(k,:)';
    v=W*x;
    y=Sigmoid(v);   %�������
    disp(y);
end


%����Ϊһ�ε�������
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

function y=Sigmoid(v)
y=1/(1+exp(-v));
end
