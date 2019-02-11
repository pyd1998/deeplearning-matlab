%单层神经网络，随机梯度下降
%以下为测试代码
clear all;
N=4;   %dataset size
X=[0 0 1;   %input
    0 1 1;
    1 0 1;
    1 1 1];
D=[0 0 1 1]; %right output
W=2*rand(1,3)-1; %start weights
for epoch=1:100000      %进行100000轮迭代
    W=DeltaSGD(W,X,D,N);
end

for k=1:N         %最终测试
    x=X(k,:)';
    v=W*x;
    y=Sigmoid(v);   %最终输出
    disp(y);
end


%以下为一次迭代过程
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

function y=Sigmoid(v)
y=1/(1+exp(-v));
end
