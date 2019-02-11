%单层神经网络，随机梯度算法与批量算法的比较
%以下为测试代码
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
    for k=1:N    %计算误差平均值
        x=X(k,:)';
        d=D(k);%标签准确值
        v1=W1*x;
        y1=Sigmoid(v1);%随机梯度法的结果
        es1=es1+(d-y1)^2;%误差累计
        v2=W2*x;
        y2=Sigmoid(v2);%批量算法的结果
        es2=es2+(d-y2)^2;%误差累计
    end
    E1(epoch)=es1/N;%计算每一轮后的误差平均值
    E2(epoch)=es2/N;
end
plot(E1,'r');
hold on
plot(E2,'b');
xlabel('Epoch');
ylabel('训练误差平均值');
legend('SGD','Batch');

function W=DeltaSGD(W,X,D,N)
%W为权重矩阵
%X为输入矩阵
%D为正确输出
alpha=0.9;          %learning rate                
for k=1:N
    x=X(k,:)';         %选取一个输入
    d=D(k);         %对应的标签值
    v=W*x;          %前向传播
    y=Sigmoid(v);   %激活
    e=d-y;          %误差值
    delta=y*(1-y)*e;
    dw=alpha*delta*x;%广义增量公式
    W(1)=W(1)+dw(1);
    W(2)=W(2)+dw(2);
    W(3)=W(3)+dw(3);%权值更新
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
dWavg=dWsum/N; %每轮只算一次迭代
W(1)=W(1)+dWavg(1);
W(2)=W(2)+dWavg(2);
W(3)=W(3)+dWavg(3);
end

function y=Sigmoid(v)
y=1/(1+exp(-v));
end