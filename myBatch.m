%单层神经网络，批量算法
%以下为训练代码
clear all;
N=4;   %dataset size
X=[0 0 1;   %input
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 0 1 1]; %right output
W=2*rand(1,3)-1; %start weights
for epoch=1:40000      %进行100000轮迭代
    W=DeltaBatch(W,X,D,N);
end

for k=1:N         %最终测试
    x=X(k,:)';
    v=W*x;
    y=Sigmoid(v);   %最终输出
    disp(y);
end

%以下为一次迭代
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
dWavg=dWsum/N; %每轮只算一次迭代
W(1)=W(1)+dWavg(1);
W(2)=W(2)+dWavg(2);
W(3)=W(3)+dWavg(3);
end
function y=Sigmoid(v)
y=1/(1+exp(-v));
end