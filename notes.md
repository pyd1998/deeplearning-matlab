---
notes for matlab deep learning
---

# deep learning using matlab

## 单层神经网络

1. 增量规则：

$$
\omega_{ij}=\omega_{ij}+\alpha e_{i} x_{j}
$$

​    其中$e_{i}$ 为输出节点 $i$ 的误差，$\alpha$ 为学习率，$x_{j}$ 为节点$j$ 对节点$i$ 的输入。$\omega_{ij}$ 为节点$j$ 到节点$i$ 的权重。

2. 广义增量规则：
   $$
   \omega_{ij}=\omega_{ij}+\alpha \delta_{i} x_{j}
   $$
   其中$\delta_{i} = \phi' ( \nu_{i} ) e_{i}$ ,    $\nu_{i}$ 为节点$i$ 激活前的输出值。$\phi$ 为激活函数。

3. 权值更新原理：

   > 随机梯度下降法：随机选取每个数据，按照增量规则对网络参数进行权值调整。
   >
   > 批量算法：使用全部训练数据计算权重更新值，然后利用平均值来调整该权重。
   >
   > 小批量算法：每次使用部分数据的平均值。使训练过程分批进行。

4. 示例:

   > [随机梯度下降法](./mySGD.m "matlab文件")
   >
   > [批量算法](./myBatch.m  "matlab文件")
   >
   > [二者比较](./SGD_BATCH.m "matlab文件")

## 多层神经网络

## 神经网络分类

## 深度学习

## 卷积神经网络