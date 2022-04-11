> 写在前面
>
> - 在此文档中的矩阵书写中，是按照Matlab的代码书写规则而来，逗号(,)代表两边相邻的元素是同一行的，分号(;)代表两边相邻的元素是分行的。
> - 注意上标和下标的区分，上标加圆括号代表的是第几个训练样本的对应数值，上标加方括号代表的是网络不同的层，下标代表的是当前参数的第几个维度。
> - 

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
- $X = [x^{(1)},x^{(2)},x^{(3)},......x^{(m)}]$ ： 这里的每个$x^{(i)}$都是一个$n_x*1$的列向量为$n_x * m$ 维矩阵
- $Y = [y^{(1)},y^{(2)},y^{(3)},......y^{(m)}]$ : 为$1*m$的矩阵

### 2. Logistic Regression

- $\hat{y} = P(y=1|x)$   表示对真实值的估计,==取值应该在0和1之间==，理解为是预测值

- $\sigma(x)$ 是 $sigmoid$ 函数的简写  
  $$
  \sigma(z) = \frac{1}{1+e^{-z}} \tag{2.1.2.1}
  $$

- $\hat{y} = \sigma(w^Tx + b$)    $w$是特征前的系数，$b$是偏置量
- 定义$x_0 = 1$，则$x^{(i)}$ 为 $(n_x + 1)*1$维度
- 定义$\theta = [\theta^0;\theta^1;\theta^2;......\theta^{n_x}]$ 其中$\theta^0 = b,\theta^{(i)} = w^{(i)}, i≥1$
- 那么$\hat{y} = \sigma(\theta^Tx)$

### 3. Cost Function and Loss(Error) Function

- 一种定义Loss
  $$
  L(\hat{y},y) = \frac{1}{2}(\hat{y}-y)^2\tag{2.1.3.1}
  $$

  > 这种定义方法在后期的优化问题中会变成非凸的（non-convex）,会得到许多局部最优解，而不是全局最优解吗，梯度下降法就不能用了

- 另一种定义Loss，对于Logistic Regression 而言
  $$
  L(\hat{y},y) = -(ylog(\hat{y})+(1-y)log(1-\hat{y}))\tag{2.1.3.2}
  $$

- 定义Cost Function
  $$
  J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)},y^{(i)})\tag{2.1.3.3}
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

### 5. Logistic Regression Derivatives 求逻辑回归的导数

- 由[2. Logistic Regression](#2. Logistic Regression)和[3. Cost Function and Loss(Error) Function](#3. Cost Function and Loss(Error) Function)得到如下公式：
  $$
  \sigma(z) = \frac{1}{1+e^{-z}} \tag{2.1.5.1}
  $$
  
  $$
  z = w^Tx+b \tag{2.1.5.2}
  $$
  
  
  $$
  \hat{y} = \sigma(z)\tag{2.1.5.3}
  $$
  
  $$
  L(\hat{y},y) = -(ylog(\hat{y})+(1-y)log(1-\hat{y}))\tag{2.1.5.4}
  $$
  
  $$
  J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)},y^{(i)})\tag{2.1.5.5}
  $$
  
- 根据[4.Gradient Descent 梯度下降法](#4. Gradient Descent 梯度下降法)中提到的代码规范，先以$Loss Function$ 为$FinalOutput$求各级变量的导数

  - 根据公式$2.1.5.4$，其中$y$是给定label，是已知的常数，而$\hat y$是受到变量参数$\theta$ 影响的变量。
    $$
    \frac{dL}{d\hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} \tag{2.1.5.6}
    $$
    注意：公式$2.1.5.4$中的$log$ 是以自然常数$e$为底
    $$
    \frac{d(log_ax)}{dx} = \lim_{\Delta{x} \to 0} \frac{log_a(x+\Delta x)-log_ax}{\Delta x} \\= \lim_{\Delta{x} \to 0}\frac{1}{\Delta x} log_a(\frac{x+\Delta x}{x})\\
    = \lim_{\Delta{x} \to 0}\frac{1}{\Delta x} log_a(1+\frac{\Delta x}{x})\\
    = \lim_{\Delta{x} \to 0}log_a(1+\frac{\Delta x}{x})^\frac{1}{\Delta x}\\
    = \lim_{\Delta{x} \to 0}log_a(1+\frac{\Delta x}{x})^{\frac{x}{\Delta x}*\frac{1}{x}}\\
    = \lim_{\Delta{x} \to 0} log_ae^\frac{1}{x}\\
    = \lim_{\Delta{x} \to 0} \frac{ln e^\frac{1}{x}}{lna}\\
    = \frac{1}{xlna}
    $$

  - 根据公式$2.1.5.3$，求$\frac{dL}{dz}$ 即`dz` 
    $$
    \frac{dL}{dz} = \frac{dL}{d\hat y}\frac{d\hat y}{dz}\\
    \frac{d\hat y}{dz} = \frac{e^{-z}}{1+e^{-z}}\\
    $$

    $$
    \frac{dL}{dz} = (-\frac{y}{\hat y} + \frac{1-y}{1-\hat y})*\frac{e^{-z}}{1+e^{-z}}\\
    = (\frac{\hat y - y}{\hat y (1-\hat y)})*\frac{e^{-z}}{1+e^{-z}}\\
    = \hat{y}-y \tag{2.1.5.7}
    $$

  - 根据$2.1.5.2$ 求 $\frac{\part L}{\part w_i}$，即`dwi`
    $$
    \frac{\part L}{\part w_i} = x_i*\frac{dL}{dz}\tag{2.1.5.8}
    $$

  - 因为$b = w_0,x_0 = 1$
    $$
    \frac{dL}{db} = \frac{dL}{dz}\tag{2.1.5.9}
    $$
    

  

### 6. Logistic Regression Gradient Descent流程

- 根据参数$\theta$和输入$X$构建单个样本的损失函数$L(\hat y,y)$，这个过程叫做前向传播步骤(before propagation steps)
- 接下来就是要利用Gradient Descent 进行对参数$\theta$ 进行调整从而使$L(\hat y, y)$ 降到最低
- 由生成的单样本损失函数计算$\theta$ 每个维度的值的偏导数，这个叫做后向传播(backwards propagation steps)
- 根据代码规范，求出对应的`dwi,db`
- 对$m$个训练examples进行做同样的操作，每一个$dw_i$，这里的$w_i$是$\theta$ 的每个维度，也是每一个example的多个参数，所以对于一个确定的$i$，会有$m$个$dw_i$，然后对$dw_i$ 求和取平均
- 更新对应的参数`wi = wi - alpha*dwi;b = b-alpha*db`

> **显然在上述步骤会用到至少两个For 循环，其中一个是对$m$个example 进行遍历计算，同时对于任一个example都会要对其中的所有$dw_i$进行计算，所以这个可以拜托显式的For 循环，采取向量进行加速运算。**
>
> **深度学习中应用向量化有助于在大数据集进行简化运算。**

## Week2.2 Python and Vectorization

### 1.Vectorizing Logistic Regression 

- 不采取显式循环，仅使用向量运算实现

- 根据显式循环的做法
  $$
  z^{(i)} = w^Tx^{(i)}+b \tag{2.2.1.1}
  $$

  $$
  a^{(i)} =\hat y^{(i)} = \sigma{(z^{(i)})} \tag{2.2.1.2}
  $$

  其中$a^{(i)}$​ 也就是上文提到的 $\hat{y}^{(i)}$​

- 在第一部分[Binary Classification](#1. Binary Classification) 中提到的 $X = [x^{(1)},x^{(2)},x^{(3)},......x^{(m)}]$ 这里的$x^{(i)}$都是$n_x*1$的向量，组成的$X$ 矩阵是 $n_x*m$的矩阵

- $w$是$n_x * 1$ 的列向量

- 由公式$2.2.1.1$ 对公式进行向量化
  $$
  Z = w^TX+b \tag{2.2.1.3}
  $$
  其中$Z = [z^{(1)},z^{(2)},z^{(3)},z^{(4)},......z^{(m)}]$

- 由公式$2.2.1.2$对公式进行向量化
  $$
  A = \sigma(Z) \tag{2.2.1.4}
  $$
  其中$A = [a^{(1)},a^{(2)},a^{(3)},a^{(4)},......a^{(m)}]$

```python
Z = np.dot(w.T,x)+b
A = sigmoid(Z)
```



### 2. Vectorizing Logistic Regression's Gradient Output

- 根据[1.Vectorizing Logistic Regression ](1.Vectorizing Logistic Regression )利用向量化对m个训练数据进行计算梯度

  > 注意上标和下标的区分，上标代表的是第几个example的对应数值，下标代表的是当前参数的第几个维度。
  >
  > 一个样例有一个$y$，计算出一个$a$，计算出一个$dz$，但有多个$w_i$

- 根据单数据梯度[5. Logistic Regression Derivatives 求逻辑回归的导数](#5. Logistic Regression Derivatives 求逻辑回归的导数)中公式$2.1.5.7$和[4.Gradient Descent 梯度下降法](#4. Gradient Descent 梯度下降法)中提到的代码规范，可得
  $$
  dz^{(1)} = a^{(1)} - y^{(1)}\\
  dz^{(2)} = a^{(2)} - y^{(2)}\\......\\
  dz^{(i)} = a^{(i)} - y^{(i)}\\
  $$
  对其进行向量化
  $$
  dZ = [dz^{(1)},dz^{(2)},.....dz^{(m)}]\\
  A = [a^{(1)},a^{(2)},.....a^{(m)}],Y = [y^{(1)},y^{(2)},.....y^{(m)}]\\
  $$

  $$
  dZ = A-Y \tag{2.2.2.1}
  $$

  

- 根据公式$2.1.5.8$进行向量化，对于一个example可以把所有的`dw_i`构成一个向量
  $$
  dw^{(i)} = [\frac{\part L}{\part w_1},\frac{\part L}{\part w_2},.....\frac{\part L}{\part w_{n_x}}]^T \tag{2.2.2.2}
  $$
  而对于m个example
  $$
  dw =\frac{1}{m} \sum_{i = 1}^{m} dw^{(i)}\tag{2.2.2.3}
  $$
  最终所需要的就是`dw`

- 根据公式$2.1.5.8$、$2.2.2.1$、$2.2.2.3$可以推出`dw`的求法

  - 首先$dw$ 是一个$n_x * 1$的向量，根据公式$2.2.2.3$可得
    $$
    dw_i =\frac{1}{m} \sum_{j=1}^{m} dw_i^{(j)} \tag{2.2.2.4}
    $$
    根据公式$2.1.5.8$可得
    $$
    dw_i^{(j)} =x_i^{(j)}*dz^{(j)}\tag{2.2.2.5}
    $$

    > 一个example 对应一个$dz$，而一个example对应的输入$x$是一个$n_x*1$的向量

    可以看出$x_i^{(j)}$和$dz^{(j)}$中都含有$j$，所以在对$j$ from 1 to m 时两者都是变量。所以只有当$x_i^{(j)}$ 构成行向量，$dz^{(j)}$ 构成列向量时进行运算才可以实现这样的结果。即：
    $$
    x_i = [x_i^{(1)},x_i^{(2)},x_i^{(3)},.....,x_i^{(m)}]\tag{2.2.2.6}
    $$

    $$
    dz = [dz^{(1)},dz^{(2)},dz^{(3)},......dz^{(m)}]^T\tag{2.2.2.7}
    $$

    $$
    dw_i =\frac{1}{m} x_idz\tag{2.2.2.8}
    $$

  - 因为$dw$是$n_x*1$的列向量，所以$dw_i$独占一行。
    $$
    dw = [dw_1;dw2;dw3;......dw_{n_x}]\tag{2.2.2.9}
    $$
    由公式$2.2.2.8$ 发现只有$x_i$和$i$ 有关，根据公式$(2.2.2.6)$可知$x_i$是一个行向量，所以令每一个$x_i$占一行。
    $$
    x = [x_1;x_2;x_3;......x_m] \tag{2.2.2.10}
    $$

  - 根据公式$2.2.2.8,2.2.2.9,2.2.2.10$ ，每一个$dw_i$组成最终的$dw$，可得
    $$
    dw = \frac{1}{m}xdz
    $$
    而会发现
    $$
    dz = dZ^T\\
    x = X
    $$
    所以综上

  $$
  dw =\frac{1}{m}XdZ^T \tag{2.2.2.11}
  $$

- 根据公式$2.1.5.9$可得m个example的db和
  $$
  db = \frac{1}{m}\sum_{i=1}^{m} dz^{(i)}\tag{2.2.2.12}
  $$

  ```python 
  db = np.sum(dZ)/m
  ```

- python代码实现

  ```python
  Z = np.dot(w.T,X)+b
  A = sigmoid(Z)
  dZ = A-Y
  dw = np.dot(X,dZ.T)/m
  db = np.sum(dZ)/m
  # 更新迭代参数
  w = w - alpha*dw
  b = b - alpha*db
  ```

  



## Week3 Shallow Neural Network

### 1. Overview

<img src="\Images\image-20220406141719813.png" alt="image-20220406141719813" style="zoom:33%;" />

- 神经网络中每一个节点都代表的是一个基于上层输入($X$)，在当前层的参数($w,b \ or \ \theta$)下，经过激活函数的运算（可能是sigmoid($\sigma(z)$)也可能是ReLU）后的一个Logistics Regression，所以遵循上文中提到的Logistics Regression中的前向传播和后向传播运算。

- 输入层(Input Layer)：开始的输入($x_1,x_2,x_3......$)列

- 输出层(Output Layer)：最后只有一个节点的层

- 隐藏层(Hidden Layer)：除去输入层和输出层后的中间的若干层

  > 在训练过程中的训练集中，我们只知道输入和输出的数值，并不会知道内部隐藏层每一层的具体的数值。

- 用$a^{[i]}$来表示第$i$层所有数值组成的列向量
  $$
  a^{[i]} = [a^{[i]}_1,a^{[i]}_2,a^{[i]}_3,a^{[i]}_4,......]^T \tag{3.1.1}
  $$

  $$
  a^{[0]} = x \tag{3.1.2}
  $$

- 所以按照第一点的步骤，当前层$a^{[i]}$是基于$a^{[i-1]}$经过激活函数的计算得到的



### 2. Neural Network 详细解析

<img src="\Images\image-20220406144108442.png" alt="image-20220406144108442" style="zoom:33%;" />

- 如上图，每一个节点内部可以细化成两个计算部分（以sigmoid函数为激活函数），总结如下
  $$
  z^{[i]}_j = (w^{[i]}_j)^Ta^{[i-1]} + b^{[i]}_j \tag{3.2.1}
  $$

  $$
  a^{[i]}_j = \sigma(z^{[i]}_j) \tag{3.2.2}
  $$

  $$
  a^{[i]} = [a^{[i]}_1,a^{[i]}_2,a^{[i]}_3,.....]^T \tag{3.2.3}
  $$

  每一层的每一个计算单元都有一个$w^{[i]}_j$和$b^{[i]}_j$ 组成这个计算单元，也可以认为是每个计算单元的属性值。

-  公式矩阵化

  - $(w^{[i]}_j)^T$ 是$1*len(a^{[i-1]})$的行向量，$a^{[i-1]}$是$len(a^{[i-1]})*1$的列向量，其中$len(a^{[i]})$表示第$i$层的节点个数。

  - 因为结果$a^{[i]}$ 是$len(a^{[i]})*1$的列向量，对应的$z^{[i]}$也是$len(a^{[i]})*1$的列向量，所以根据公式$(3.2.1)$可以得出以下内容。
  
  - 因为$1 \leq j \leq len(a^{[i]})$，将$(w^{[i]}_j)^T$ 作为$W^{[i]}$的**每一行**，将$b^{[i]}_j$作为$b^{[i]}$的每一个行。
    $$
    W^{[i]} = [(w^{[i]}_1)^T;(w^{[i]}_2)^T;(w^{[i]}_3)^T;......]
    $$
    $$
    b^{[i]} = [b^{[i]}_1,b^{[i]}_2,b^{[i]}_3,......]
    $$
    
    所以$W^{[i]}$是$len(a^{[i]})*len(a^{[i-1]})$​的，$b^{[i]}$是$len(a^{[i]}*1)$的列向量。
    
    因为$a^{[i-1]}$是$len(a^{[i-1]})*1$的，根据矩阵相乘，得到的结果$z^{[i]}$ 就是$len(a^{[i]})*1$的。
  
  - 综上：
    $$
    z^{[i]} = W^{[i]}a^{[i-1]} + b^{[i]} \tag{3.2.4}
    $$
  
    $$
    a^{[i]} = \sigma(z^{[i]}) \tag{3.2.5}
    $$
  
  

### 3. 多样本 向量化(forward propagation)

当存在$m$个样本时，朴素的方法是通过循环进行对每个样本进行计算，但这不是最好的方法。

- 对于第$j$个样本，在网络的第$i$层时的计算公式为：
  $$
  z^{[i](j)} = W^{[i]}a^{[i-1](j)} + b^{[i]} \tag{3.3.1}
  $$

  $$
  a^{[i](j)} = \sigma(z^{[i](j)}) \tag{3.3.2}
  $$

  > 注意这里的圆括号和方括号的区别，圆括号代表的是第几个样本，方括号代表的是该样本计算在网络中的第几层。
  >
  > 同时注意因为$W^{[i]}$和$b^{[i]}$是每个网络的每一层专有的属性值，不会随训练样本的变化而变化，所以上标没有必要加$(j)$

- 回顾[Binary Classification](# 1. Binary Classification)可以得到

  - $X$是$n_x*m$的矩阵，也就是$len(a^{[0]})*m$么，每一列代表着是一个训练样本的数据。

  - $W^{[1]}$是$len(a^{[1]})*len(a^{[0]})$的矩阵，每一行都是第一层的对应节点的参数

  - 所以参数和数据要对应相乘，即行\*列，所以$W^{[1]}X$，结果为$len(a^{[1]})*m$，再加上$b^{[1]}$，这里要利用python广播的原理，**每一列是对应的是某一数据的在第一层计算结果，每一行代表的是所有数据经过了第一层中某一节点的计算结果。**

  - $A^{[i]}$和$Z^{[i]}$都是$len(a^{[i]})*m$的。

  - 所以综上:
    $$
    Z^{[1]} = W^{[1]}X+b^{[1]}
    $$

    $$
    A^{[1]} = \sigma(Z^{[1]})
    $$

    扩展到一般化

  $$
  Z^{[i]} = W^{[i]}A^{[i-1]}+b^{[i]} \tag{3.3.3}
  $$

  $$
  A^{[i]} = \sigma(Z^{[i]})\tag{3.3.4}
  $$

  $$
  A^{[0]} = X \tag{3.3.5}
  $$

   

### 4. Activation Function 激活函数

- 双曲函数

  - 双曲正弦
    $$
    sinhx = \frac{e^x-e^{-x}}{2}
    $$

  - 双曲余弦
    $$
    coshx = \frac{e^x+e^{-x}}{2}
    $$

  - **双曲正切**
    $$
    tanhx = \frac{sinhx}{coshx}=\frac{e^x-e^{-x}}{e^x+e^{-x}} \tag{3.4.1}
    $$

  - 双曲余切
    $$
    cothx = \frac{1}{tanhx}=\frac{e^x+e^{-x}}{e^x-e^{-x}}
    $$

  - 双曲正割
    $$
    sechx = \frac{1}{coshx} = \frac{2}{e^x+e^{-x}}
    $$
    
  - 双曲余割
    $$
    cschx = \frac{1}{sinhx} =\frac{2}{e^x-e^{-x}}
    $$
  
  - 目前基本不用$\sigma()$作为激活函数了，相对于$\sigma()$而言，$tanh()$更加优越，因为激活函数的平均值更接近0，有助于数据中心化。
  
  - 但对于二元分类的情况下，希望输出值是0和1，这个时候一般在输出层会使用$\sigma()$作为激活函数。
  
  - 所以对于一个网络，不同层之间的激活函数是是可以不一样的，可以再激活函数上加上方括号角标以作区分。
  
  - $$
    \frac{d(tanhx)}{dx} = 1-(tanhx)^2 \tag{3.4.2}
    $$
  
    
  
- ReLU(rectified linear unit)
  $$
  a =  max(0,z)
  $$
   通常使用ReLU，会使神经网络学习速度快很多

  

- 隐藏单元不能使用线性的激活函数，必须使用ReLU 或tanh



### 5. 神经网络的梯度下降法(back propagation)

这里依旧是二分类算法，所以根据前文的内容可以总结出一下公式

- [forward propagation](#3. 多样本 向量化(forward propagation))的公式

$$
Z^{[i]} = W^{[i]}A^{[i-1]}+b^{[i]} \tag{3.3.3}
$$

$$
A^{[i]} = \sigma(Z^{[i]})\tag{3.3.4}
$$

$$
A^{[0]} = X \tag{3.3.5}
$$

- 设各层网络的单元数为$n^{[i]}$，即
  $$
  len(a^{[i]}) = n^{[i]} \tag{3.5.1}
  $$

  $$
  n^{[0]} = n_x
  $$

- 根据[3. Cost Function and Loss(Error) Function](#3. Cost Function and Loss(Error) Function)，得到公式$(2.1.3.2),(2.1.3.3)$
  $$
  L(\hat{y},y) = -(ylog(\hat{y})+(1-y)log(1-\hat{y}))\tag{2.1.3.2}
  $$
  
  $$
  J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)},y^{(i)})\tag{2.1.3.3}
  $$

- 神经网络的参数为$w^{[i]},b^{[i]}$

- 根据[2. Vectorizing Logistic Regression's Gradient Output](# 2. Vectorizing Logistic Regression's Gradient Output)，得到公式
  $$
  dZ = A-Y \tag{2.2.2.1}
  $$

  $$
  dw =\frac{1}{m}XdZ^T \tag{2.2.2.11}
  $$

  $$
  db = \frac{1}{m}\sum_{i=1}^{m} dz^{(i)}\tag{2.2.2.12}
  $$
  
  > 分析：
  >
  > 对于单节点（单元）的公式，其中A是一个$1*m$的**行向量**，每一个值都代表着某一个样本经过该神经元后的结果。
  >
  > 而在神经网络中的$A^{[i]}$，因为每一层有多个神经元，$A^{[i]}$中的每一行都代表着一个神经元，即上文中提到的**每一列是对应的是某一个数据样本的在该层计算结果，每一行代表的是所有的数据样本经过了该层中某一节点的计算结果。**
  >
  > 前一层的神经单元数目$n^{[i-1]}$ 就是下一层每个神经元的输入列向量的维数。
  
- 在此基础上结合公式$(3.3.3),(3.3.4),(3.3.5)$得到Back Propagation。假设网络的总共有$k$层，最后一层只有一个神经单元，所以$A^{[k]}$只有一行，符合数据label:$Y$也是一行的要求。
  $$
  dZ^{[k]} = A^{[k]}-Y \tag{3.5.2}
  $$

  值得注意的是这里和上文中的计算不一样，按照$(2.2.2.11)$的格式应该如下：
  $$
  dw^{[k]} =\frac{1}{m}A^{[k-1]}(dZ^{[k]})^T \tag{3.5.3}
  $$

  但是我们来验证一下是否正确

  首先看一下原公式$(2.2.2.11)$
  $$
  dw =\frac{1}{m}XdZ^T \tag{2.2.2.11}
  $$
  当只有一个神经单元时，$dw$是一个$n_x*1$的列向量，$X$是一个$n_x*m$的矩阵（其中==每一列代表的是一个数据样本==，共有$m$列）$dZ$是一个由$dz$组成的$1*m$的行向量（每一个测试样本都对应的一个$dz$），转置之后就是$m*1$的列向量。

  其次我们关注一下公式$(2.2.2.6),(2.2.2.7),(2.2.2.8)$

  
  $$
  x_i = [x_i^{(1)},x_i^{(2)},x_i^{(3)},.....,x_i^{(m)}]\tag{2.2.2.6}
  $$

  $$
  dz = [dz^{(1)},dz^{(2)},dz^{(3)},......dz^{(m)}]^T\tag{2.2.2.7}
  $$

  $$
  dw_i =\frac{1}{m} x_idz\tag{2.2.2.8}
  $$

  ==这了跟$dz$依次相乘的是每一个样本列向量的同一个位置（维度）组成的行向量。==

  所以我们再看公式$(3.5.3)$是怎么构成的

  - $A^{[k-1]}$代表的是第$k-1$层（倒数第二层）的计算结果，是一个$n^{[k-1]}*m$的矩阵（每一列代表着一个训练样本经过前$k-1$层网络得出的结果，其中的每一行的数代表经过第$k-1$层的某一个节点后的结果）
  
  - $(dZ^{[k]})^T$由公式$(3.5.2)$可知，是一个$m*1$的列向量
  
  - 但是看一下两者相乘的意义：相乘得到应该是一个$n^{[k-1]}*1$的列向量，列向量的每一行的值都是由“每一个训练样本经过$k-1$层中的某一个节点后的值”和“每一个训练样本对应的$dz$”分别相乘再相加的结果，再取均值，那么说明这个结果就是这个节点的$dw$
  
  - 但是在后面的$W^{[k]}$参数更新时，要实现公式
    $$
    W^{[k]} = W^{[k]}-\alpha dw \tag{3.5.4}
    $$
    这里的$W^{[k]}$的**每一行**对应的是第$k$层的一个节点的参数，所以这里的$W^{[k]}$是一个$1*n^{[k-1]}$的行向量，刚好和$dw$的转置维度相等。所以其实这里需要的是这个$dw^{[k]}$的转置，所以真正的$dw^{[k]}$应该如下：
    $$
    dw^{[k]} =\frac{1}{m}dZ^{[k]}(A^{[k-1]})^T \tag{3.5.5}
    $$
    





