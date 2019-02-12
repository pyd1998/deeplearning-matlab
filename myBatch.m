%���������磬�����㷨
%����Ϊѵ������
clear all;
N=4;   %dataset size
X=[0 0 1;   %input
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 0 1 1]; %right output
W=2*rand(1,3)-1; %start weights
for epoch=1:40000      %����100000�ֵ���
    W=DeltaBatch(W,X,D,N);
end

for k=1:N         %���ղ���
    x=X(k,:)';
    v=W*x;
    y=Sigmoid(v);   %�������
    disp(y);
end

%����Ϊһ�ε���
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