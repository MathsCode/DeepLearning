## Week1 Introduction to Deep Learning

### 1. 神经网络中的一个巨大突破：从Sigmoid 函数到 Relu函数 的迁移

- sigmoid 函数 在梯度近乎为0的区域学习的进度会非常缓慢，因为使用的是梯度下降法，参数变化的速率非常慢。
- 当把激活函数换成ReLU（rectified linear unit）,对于所有正输入梯度都是1那么梯度就不会慢慢变成 0。 那么梯度就不会慢慢变成 0。 梯度在左边的时候为 0。
- 结果证明， 简单地把 sigmoid 函数换成 ReLU 函数 使得梯度下降算法的速度 提高很多。

## Week2.1 Logistic Regression as a Neural Network

### 1. Binary Classification

- $m$ :  training examples 等价于$m_{train}$
- $m_{test}$ : test examples
- $n_x$ : 表示输入的特征个数，即x的维数
- $X = [x^{(1)},x^{(2)},x^{(3)},......x^{(m)}]$ ： 为$n_x * m$ 维矩阵
- $Y = [y^{(1)},y^{(2)},y^{(3)},......y^{(m)}]$ : 为$1*m$的矩阵

### 2. Logistic Regression

- $\hat{y} = P(y=1|x)$   表示对真实值的估计,==取值应该在0和1之间==

- $\sigma(x)$ 是 $sigmoid$ 函数的简写  
  $$
  \sigma(z) = \frac{1}{1+e^{-z}}
  $$

- $\hat{y} = \sigma(w^Tx + b$)    $w$是特征前的系数，$b$是偏置量
- 定义$x_0 = 1$，则$x^{(i)}$ 为 $(n_x + 1)*1$维度
- 定义$\theta = [\theta^0;\theta^1;\theta^2;......\theta^{n_x}]$ 其中$\theta^0 = b,\theta^{(i)} = w^{(i)}, i≥1$
- 那么$\hat{y} = \sigma(\theta^Tx)$

### 3. Cost Function



