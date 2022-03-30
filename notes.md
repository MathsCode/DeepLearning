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

- $\hat{y} = P(y=1|x)$   表示对真实值的估计,==取值应该在0和1之间==，理解为是预测值

- $\sigma(x)$ 是 $sigmoid$ 函数的简写  
  $$
  \sigma(z) = \frac{1}{1+e^{-z}}
  $$

- $\hat{y} = \sigma(w^Tx + b$)    $w$是特征前的系数，$b$是偏置量
- 定义$x_0 = 1$，则$x^{(i)}$ 为 $(n_x + 1)*1$维度
- 定义$\theta = [\theta^0;\theta^1;\theta^2;......\theta^{n_x}]$ 其中$\theta^0 = b,\theta^{(i)} = w^{(i)}, i≥1$
- 那么$\hat{y} = \sigma(\theta^Tx)$

### 3. Cost Function and Loss(Error) Function

- 一种定义Loss
  $$
  L(\hat{y},y) = \frac{1}{2}(\hat{y}-y)^2
  $$

  > 这种定义方法在后期的优化问题中会变成非凸的（non-convex）,会得到许多局部最优解，而不是全局最优解吗，梯度下降法就不能用了

- 另一种定义Loss，对于Logistic Regression 而言
  $$
  L(\hat{y},y) = -(ylog(\hat{y})+(1-y)log(1-\hat{y}))
  $$

- 定义Cost Function
  $$
  J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)},y^{(i)})
  $$
  

> Loss Function 是针对单个训练示例定义的损失函数，衡量的是在单个训练示例中Hypothesis的表现。
>
> Cost Function 是针对整个训练集定义的成本函数，衡量的是在整个训练集的中Hypothesis的表现

### 4. Gradient Descent 梯度下降法

> 梯度下降法会使函数向最优解的方向前进
>
> 在编程中的代码规范
>
> - $\frac{dFinalOutput}{dvar}$ 或者 $\frac{\partial FinalOutput}{\partial var}$表示最终输出对某一变量的求导，一般代码中写成`dvar`即代表。

### 5. Logistic Regression Gradient Descent

- 根据参数$\theta$和输入$X$构建单个样本的损失函数$L(\hat y,y)$，这个过程叫做前向传播步骤(before propagation steps)
- 接下来就是要利用Gradient Descent 进行对参数$\theta$ 进行调整从而使$L(\hat y, y)$ 降到最低
- 由生成的单样本损失函数计算$\theta$ 每个维度的值的偏导数，这个叫做后向传播(backwards propagation steps)
- 根据代码规范，求出对应的`dwi,db`
- 更新对应的参数`wi = wi - alpha*dwi;b = b-alpha*db`

