%����Ϊѵ������
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
X=Images(:,:,8001:10000);%���ղ���
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
%����Ϊһ�ε�������
function [W1,W5,Wo]=MnistConv(W1,W5,Wo,X,D)
%W1Ϊ�����Ȩֵ��W5Ϊ���ز�Ȩֵ��WoΪ�����Ȩֵ
alpha=0.01;
beta=0.95;%������ϵ��
momentum1=zeros(size(W1));
momentum5=zeros(size(W5));
momentumo=zeros(size(Wo));
N=length(D);
bsize=100;  %С�����㷨��batchsize=100
blist=1:bsize:(N-bsize+1);%�洢ÿһ�����׸����ݵ�����
for batch=1:length(blist)%��С��ѵ��
    dw1=zeros(size(W1));
    dw5=zeros(size(W5));
    dwo=zeros(size(Wo));
    begin=blist(batch);%ѡ��һ������������
    for k=begin:begin+bsize-1%ѡ��һ������
        x=X(:,:,k);%����
        d=zeros(10,1);
        d(sub2ind(size(d),D(k),1))=1;%��������������ʾ
        y1=Conv(x,W1);%���
        y2=ReLU(y1);%����
        y3=Pool(y2);%�ػ�
        y4=reshape(y3,[],1);%ת��������������ȫ���Ӳ�
        v5=W5*y4;
        y5=ReLU(v5);
        v=Wo*y5;
        y=Softmax(v);%ȫ�����е�ǰ�����
        %
        e=d-y;
        delta=e;
        e5=Wo'*delta;%���򴫲�
        delta5=(v5>0).*e5;%���ز�
        e4=W5'*delta5;%�ػ���
        e3=reshape(e4,size(y3));%��Ȼ�����£�������Ҫ��������
        e2=zeros(size(y2));
        W3=ones(size(y2))/(2*2);
        for c=1:20
            e2(:,:,c)=kron(e3(:,:,c),ones([2,2])).*W3(:,:,c);
        end
        delta2=(y1>0).*e2;%ReLU��
        delta1_x=zeros(size(W1));%�����
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