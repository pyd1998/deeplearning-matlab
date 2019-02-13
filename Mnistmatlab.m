%以下为训练代码
clear all;
Images=loadMNISTImages('./MNIST/t10k-images.idx3-ubyte');
Images=reshape(Images,28,28,[]);
Labels=loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte');
Labels=Labels+10*(Labels==0);
rng(1);
W1=1e-2*randn([9,9,20]);
W5=(2*rand(100,2000)-1)*sqrt(6)/sqrt(360+2000);
Wo=(2*rand(10,100)-1)*sqrt(6)/sqrt(10+100);
X=Images(:,:,1:8000);
D=Labels(1:8000);
for epoch=1:3
    disp(epoch);
    [W1,W5,Wo]=MnistConv(W1,W5,Wo,X,D);
end
save('MnistConv.mat');
X=Images(:,:,8001:10000);%最终测试
D=Labels(8001:10000);
acc=0;
N=length(D);
for k=1:N
    x=X(:,:,k);
    y1=Conv(x,W1);
    y2=ReLU(y1);
    y3=Pool(y2);
    y4=reshape(y3,[],1);
    v5=W5*y4;
    y5=ReLU(v5);
    v=Wo*y5;
    y=Softmax(v);
    [~,i]=max(y);
    if i==D(k)
        acc=acc+1;
    end
end
acc=acc/N;
disp(acc);
%以下为一次迭代过程
function [W1,W5,Wo]=MnistConv(W1,W5,Wo,X,D)
%W1为卷积层权值，W5为隐藏层权值，Wo为输出层权值
alpha=0.01;
beta=0.95;%动量法系数
momentum1=zeros(size(W1));
momentum5=zeros(size(W5));
momentumo=zeros(size(Wo));
N=length(D);
bsize=100;  %小批量算法，batchsize=100
blist=1:bsize:(N-bsize+1);%存储每一批的首个数据的索引
for batch=1:length(blist)%分小批训练
    dw1=zeros(size(W1));
    dw5=zeros(size(W5));
    dwo=zeros(size(Wo));
    begin=blist(batch);%选择一个首数据索引
    for k=begin:begin+bsize-1%选择一批数据
        x=X(:,:,k);%输入
        d=zeros(10,1);
        d(sub2ind(size(d),D(k),1))=1;%将数字用向量表示
        y1=Conv(x,W1);%卷积
        y2=ReLU(y1);%激活
        y3=Pool(y2);%池化
        y4=reshape(y3,[],1);%转换成列向量输入全连接层
        v5=W5*y4;
        y5=ReLU(v5);
        v=Wo*y5;
        y=Softmax(v);%全连接中的前向过程
        %
        e=d-y;
        delta=e;
        e5=Wo'*delta;%反向传播
        delta5=(v5>0).*e5;%隐藏层
        e4=W5'*delta5;%池化层
        e3=reshape(e4,size(y3));%虽然不更新，但还是要进行误差传递
        e2=zeros(size(y2));
        W3=ones(size(y2))/(2*2);
        for c=1:20
            e2(:,:,c)=kron(e3(:,:,c),ones([2,2])).*W3(:,:,c);
        end
        delta2=(y1>0).*e2;%ReLU层
        delta1_x=zeros(size(W1));%卷积层
        for c=1:20
            delta1_x(:,:,c)=conv2(x(:,:),rot90(delta2(:,:,c),2),'valid');
        end
        dw1=dw1+delta1_x;
        dw5=dw5+delta5*y4';
        dwo=dwo+delta*y5';
    end
    dw1=dw1/bsize;dw5=dw5/bsize;dwo=dwo/bsize;
    momentum1=alpha*dw1+beta*momentum1;
    momentum5=alpha*dw5+beta*momentum5;
    momentumo=alpha*dwo+beta*momentumo;
    W1=W1+momentum1;W5=W5+momentum5;Wo=Wo+momentumo;
end
end
function y=Conv(x,W)
[wrow,wcol,numFilters]=size(W);
[xrow,xcol,~]=size(x);
yrow=xrow-wrow+1;
ycol=xcol-wcol+1;
y=zeros(yrow,ycol,numFilters);
for k=1:numFilters
    filter=W(:,:,k);
    filter=rot90(squeeze(filter),2);
    y(:,:,k)=conv2(x,filter,'valid');
end
end
function y=Pool(x)
[xrow,xcol,numFilters]=size(x);
y=zeros(xrow/2,xcol/2,numFilters);
for k=1:numFilters
    filter=ones(2)/(2*2);
    image=conv2(x(:,:,k),filter,'valid');
    y(:,:,k)=image(1:2:end,1:2:end);
end
end
function y=ReLU(v)
y=max(0,v);
end
function y=Softmax(v)
ex=exp(v);
y=ex/sum(ex);
end