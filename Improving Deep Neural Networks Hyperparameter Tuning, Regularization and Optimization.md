# [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)

## Week 1 Practical Aspects of Deep Learning

### 1. Train Dev Test sets

- 数据量小，如1000个，各占70%/0/30% or 60%/20%/20%
- 数据量大，后两者不需要很多

Train Set 训练集：对于一个网络（已经确定好了超参数），用训练集进行训练，观察是否收敛，得到对应的平均 Cost or Error,若已经收敛，并且通过训练集的存在确定了网络中的参数。

Hold-out cross validation set 简单交叉验证集 or development (dev) set：对于已经通过训练集训练好的网络，可以通过dev set 进行计算得对应的dev set error从而在若干个已经收敛的网络中选择出那个最好的网络，或者说是对多个解决问题算法进行验证，检验哪种算法更有效，或者说哪些算法相对更有效，此时已经选择好最优的一个或者多个模型。

Test set 测试集：测试机的存在主要的目的是将已经选择好的网络对其的泛化能力进行测试，观察其是否存在过拟合的问题，对选拔出来的分类器的效果进行评判，所以其实有无测试集对最终的网络效果并不会产生影响，但是如果没有测试集，那个时候验证集所起到的作用就是



如果数据集中的数据的来源不一样，即代表着数据的质量、结构都会有差别，那么要确保验证集和测试集的来自于同一个来源。

### 2. 偏差（Bias） 方差（Variance）

- 高偏差，欠拟合：

- 高方差，过拟合：

- 研究bias and variance的指标：

  - Train set error: 根据测试集错误率的高低判断是否是欠拟合
  - Dev set error：根据验证集错误率的高低判断是否是过拟合

- 处理高方差、高偏差的常用方法

  - 高偏差High bias：

    首先明确高偏差的是对于训练集而言，高偏差带来的是欠拟合，说明该网络在训练集上没有很好的拟合或者是收敛。

    所以需要做的就是更换网络的结构，比如增加相应的隐藏层或者隐藏单元，同时也可以更换更适合这个问题的算法，==或者说换一些其他的网络结构，但是这个可能有用也可能没用==。

  - 高方差High Variance：

    明确高方差是对于验证集而言，高方差带来的是过拟合，过分考虑了某个特殊数据的存在，没有考虑全局性，代表这个网络没有很好的泛化性。

    所以需要做的就是，添加更多的数据（当然这是基于能够获取到更多的数据），将部分特殊数据进行掩盖掉（优化掉）。或者==是添加正则化项==，当然更换其他网络结构很有可能会做到一箭双雕，同时解决两个问题。

  - ==通常采用更多的数据可以在不过多影响偏差的同时降低方差==

### 3. 正则化（Regularization）（$L_2$ 正则化）

- 旨在解决网络过度拟合的数据，产生了高方差High variance的问题。

- $L_2 \ Regularization: $
  $$
  J(w,b) = \frac{1}{m}\sum_{i=1}^{m} L(\hat y ^{(i)},y) + \frac{\lambda}{2m}\Vert w\Vert_2^2 \tag{1.3.1}
  $$
   其中$L_2$范数
  $$
  \Vert w \Vert_2^2 = \sum_{j=1}^{n_x} w_j^2 = w^Tw \tag{1.3.2}
  $$

- $L_1 Regularization$
  $$
  J(w,b) = \frac{1}{m}\sum_{i=1}^{m} L(\hat y ^{(i)},y) + \frac{\lambda}{2m}\Vert w\Vert_1 \\
  = \frac{1}{m}\sum_{i=1}^{m} L(\hat y ^{(i)},y) + \frac{\lambda}{2m}\sum_{j = 1}^{n_x}\vert w_j\vert\tag{1.3.3}
  $$
  但是$L_1$正则化会使模型变得稀疏，也就是说$w$会有很多0

### 4.为什么正则化能够预防过拟合

- $L_2$正则化的内容是在损失函数的基础上加上一个正则化项$(1.3.2)$，为什么正则化就能预防过拟合。

- 过拟合的出现，即就是因为当原本的损失函数出现过拟合的状态，可以认为此时加上的正则化系数$\lambda$很小，即当$\lambda$趋于0的状况下，会造成过拟合的状态。

- 那么再考虑当公式中的$\lambda$很大时，此时的$w$矩阵中的会有多个接近0的值，这样在网络结构上看，代表着有多个节点是无效节点，那么此时的网络就可以看成是一个层数很多即很深的但是每层的节点数很少的网络，这样就很显然就会造成欠拟合的状态，即高偏差的效果。

  > 为什么，当$w$很小的时候，会导致欠拟合、高偏差的效果？
  >
  > 当$w$很小的时候，那么以$tanh$ 为激活函数时，$w$很小即$z$就会很小，那么自变量的取值区间所处于是$tanh$函数的近似线性空间。我们知道当激活函数的线性函数的时候，不管是多深的网络或者说每层有多少个节点，最终效果都还是线性的。所以必定会存在欠拟合的情况。

- 所以必然会存在一种情况，即$\lambda$取一个特殊的值或者区间，让网络效果处于“just right”的状态，也就是刚刚好的状态！

### 5. Dropout 正则化

- ==反向随机失活Inverted Dropout==

  - $l$ :层数

  - keep_prob:代表每个节点存活的概率，低于keep_prob则代表该节点需要保存，反之则失活。如果keep_prob为0.8代表节点失活的概率为0.2。也可以认为保留上层数据的概率。每一层都有一个keep_prob

  - $d^3$：是一个由0、1组成的矩阵，用来代表该层的节点是否保存，过程是先生成一个随机矩阵，其中每一个数代表的是第三层的某一个节点的数值，然后与keep_prob作比较，如果低则代表该节点保存，反之失活。

    ```python
    d3 = np.random.rand(a3.shape[0],a3.shape[1]) < keep_prob
    ```

    > 注：上文只针对网络的某一层（此处为第3层），第三层的输出结果是a3，那么a3的结构应该就是为第三层节点数*m总样本数。

  - 该层有节点存在失活，对a3结果也是有影响，所以需要对a3进行更新，过滤掉d3中为0对应在a3的元素

    ```python
    a3 = np.multiply(a3,d3) # a3 *= d3
    ```

  - 因为在计算$Z^{[4]} = W^{[4]}a^{[3]}+b^{[4]}$过程中，因为$a^{[3]}$被失活了部分值，所以为了保证$Z^{[4]}$的期望，需要除以keep_prob来理论上达到之前的效果

    ```python
    a3 /= keep_prob
    ```

    > 可以理解为，当a3有部分值被失活，那么产生的影响只有原本的影响（可以认为就是a3）的keep_prob倍，所以需要对其进行恢复。所以用除法。

    

    > 这也是dropout正则化的精髓所在，让部分节点失活消失作用，但是增强其他节点的影响。——摘自B站评论

    

    > 上文中提到的期望就可以理解为是均值，因为丢了部分的数据（失活归零），那么对于a3而言，均值成为了之前的keep_prob倍，所以需要除以keep_prob来恢复使均值基本变化不大。																					——摘自B站评论

  - 为什么要用除法，上文已经给了形象的解释，但是从数学本质出发，需要保证期望也就是均值不发生变化，那么对其的证明就有：

    > 假设一个神经元的输出激活值为`a`，在不使用dropout的情况下，其输出期望值为`a`，如果使用了dropout，神经元就可能有保留和关闭两种状态，把它看作一个离散型随机变量，它就符合概率论中的**0-1分布**，其输出激活值的期望变为 `p*a+(1-p)*0=pa`，此时若要保持期望和不使用dropout时一致，就要除以 `p`。



- 测试阶段不进行Dropout

   深度学习模型在训练时候要使用Dropout来解决可能会出现的overfitting的问题，注意这里的dropout是指inverted dropout，而不是传统的dropout，而inverted dropout的在测试的时候是不需要进行的。

  > 深度学习模型训练时候使用dropout实际上只是让部分神经元在当前训练批次以一定的概率不参与更新，这样使得每一轮迭代获得的模型都是不一样的。这个过程一定程度上==保持了不同模型之间最优参数设置==，使得训练出的每一个模型不至于太差。在预测时候，不使用dropout，但会在权重上都乘上保留概率。最终的输出可以被认为是Bagging的一种近似。

  > 传统dropout 和 inverted dropout的区别，目前主流的方法是inverted dropout：
  >
  > - 在训练阶段，对执行了dropout操作的层，其输出激活值要除以`keep_prob`，
  >
  > - 测试阶段则不执行任何操作，既不执行dropout，也不用对神经元的输出乘`keep_prob`。

  而对于比较久的网络模型，比如AlexNet，其中采取的就是传统的dropout，其实从数学本质上来看，两者的区别并不是很大。

  > 在训练阶段应用dropout时没有让神经元的输出激活值除以 `p`，因此其期望值为 `pa`，在测试阶段不用dropout，所有神经元都保留，因此其输出期望值为 `a` ，为了让测试阶段神经元的输出期望值和训练阶段保持一致（这样才能正确评估训练出的模型），就要给测试阶段的输出激活值乘上 `p` ，使其输出期望值保持为 `pa`。

  而选择Inverted-dropout:

  > 测试阶段的模型性能很重要，特别是对于上线的产品，模型已经训练好了，只要执行测试阶段的推断过程，那对于用户来说，当然是推断越快用户体验就越好了，而Inverted-dropout把保持期望一致的关键步骤转移到了训练阶段，节省了测试阶段的步骤，提升了速度。
  >
  > dropout方法里的 `keep_prob` 是一个可能需要调节的超参数，用Inverted-dropout的情况下，当你要改变 `keep_prob` 的时候，只需要修改训练阶段的代码，而测试阶段的推断代码没有用到 `keep_prob` ，就不需要修改了，降低了写错代码的概率。

- 为什么Dropout明明让一些节点失去作用，却能解决overfitting的问题？

  因为在训练过程中采取了dropout，每一个节点（也可以是Feature）都有可能在某一次训练中被失活，失活带来的代价就是输入在这个节点会消失，输出也会消失，所以对于每一个节点（Feature）的权重分配必须进行扩散才行，也就是不能把鸡蛋都放在一个笼子里，不能依赖于任何一个节点（Feature）。那么通过扩散权重达到雨露均沾会产生收缩权重的平方范数的效果。

- 为什么 括号里要说也有Feature？

  因为对于第一层的网络节点也可以设定对应的keep_prob，这代表着对于多个输入特征而言，这样会使某些特征失效。

- 对于keep_prob的选择：

  对于那些你认为相对于其他层有更大出现过拟合情况的层，他们的keep_prob可以相对设置低一些，但是当引入keep_prob后再交叉验证过程，要搜索更多的hyper-parameter

- 计算机视觉领域采取dropout很多的原因：
  - 对于图像而言，一个训练样本拥有的像素很多，但是硬件设备的算力、内存是一定的，所以需要的像素多了，处理的样本就少了，所以会引用进dropout来对每一张图片进行处理，以至于获得更多的样本。
  - 但在其他领域，如果不出现过拟合问题，那么就没有使用Dropout正则化的必要了。

### 6.其他正则化方法

#### Data augmentation  数据增广

可以对同一张图片（数据）轻微的处理再加入到数据集当中作为新的数据。

#### Early stopping 提前终止

神经网络没必要一定要训练到最后，可以提前终止。



### 7. 优化配置->加速训练：对输入特征进行归一化

对于某一个特征：
$$
\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

$$
x = x-\mu
$$

$$
\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}
$$

$$
x = x/\sigma
$$

### 8. 问题：梯度消失、梯度爆炸 data vanishing/exploding gradients

当权重大于1的时候，会出现指数型增长即梯度爆炸

当权重小于1的时候，会出现指数型降低即梯度消失

 

### 9. 梯度检验

通过使用双边误差的方法来逼近（计算）得到的导数$d\theta_{approx}$，和原本计算的导数$d\theta$进行比较。

双边误差的计算导数的公式：
$$
d\theta_{approx}[i] = \frac{J(\theta_1,\theta_2,...,\theta_i+\epsilon,...)-J(\theta_1,\theta_2,...,\theta_i-\epsilon,...)}{2\epsilon}
$$
利用欧几里得范数的定义（计算两个向量之间的欧氏距离）进行计算两者的差异：
$$
\frac{\Vert d\theta_{approx} - d\theta \Vert_2}{\Vert d\theta_{approx} \Vert_2+\Vert d\theta \Vert_2}
$$

### 10.Tips

- 不要再训练的时候使用梯度检验，只有在调试的时候才需要。
- 记住有正则化

## Week2 Optimization Algorithms

- 大数据情况下训练速度太慢

### 1. Mini-batch 梯度下降法

- 思想就是把整个数据集划分为若干个mini-batch，每一个mini-batch用`{}`来进行表示

  > 所以总结一下：
  >
  > `{}`用来表示mini-batch，即可表示为$X^{\{t\}}$为第$t$个mini-batch
  >
  > `[]`用来表示神经网络的层，表示为与神经网络某一层中的相关的属性数据，如$z^{[l]}$
  >
  > `()`表示特指某一个样例中的数据,如$x^{(i)}$

- 当对神经网络参数进行训练的时候，分mini-batch进行训练，每一个mini-batch训练的过程就跟之前的batch gradient descent流程一样，只不过数据规模变成了mini-batch的规模罢了。

- 与之前的batch gradient descent相比较，对整个数据集进行一次梯度下降运算就包含了整个数据集的数据，**也就是说，遍历一次数据集只能梯度下降一次，**而要得到最终的结果，需要在梯度下降的外面再套一层迭代循环，才能得到最终的最优结果。显然这样对大数据集是不友好的。

- 而对于mini-batch而言，对整个数据集进行一次遍历梯度运算，相当于做了mini-batch的个数次的梯度下降。**每一次mini-batch gradient descent 也叫做一次epoch**

- 如果有个丢失数据集，mini-batch效果更好，运行速度更快。

### 2. Mini-batch的理解

- 如何选择mini-batch的size

  - size 为m的话，就跟普通的Batch gradient descent 一样的，每次迭代需要处理大量样本，如果样本数量巨大，单词迭代时间太长。
  
  - size 为1的话，每一个样本都是一个mini-batch，叫做Stochastic Gradient Descent 随机梯度下降法，每次迭代都是一个训练样本，无法通过向量化计算来进行运算加速。
  
    > 一种变体是随机梯度下降（SGD），它相当于小批量梯度下降，每个小批量只有一个示例。您刚刚实现的更新规则不会更改。改变的是，一次只计算一个训练示例的梯度，而不是整个训练集的梯度。
  
  - 所以真正的size应该介于两者之间
  
- **(Batch) Gradient Descent**:

``` python
X = data_input
Y = labels
m = X.shape[1]  # Number of training examples
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost
    cost_total = compute_cost(a, Y)  # Cost for m training examples
    # Backward propagation
    grads = backward_propagation(a, caches, parameters)
    # Update parameters
    parameters = update_parameters(parameters, grads)
    # Compute average cost
    cost_avg = cost_total / m
        
```

- **Stochastic Gradient Descent**:

```python
X = data_input
Y = labels
m = X.shape[1]  # Number of training examples
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    cost_total = 0
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost_total += compute_cost(a, Y[:,j])  # Cost for one training example
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters
        parameters = update_parameters(parameters, grads)
    # Compute average cost
    cost_avg = cost_total / m
```



### 3. Exponentially weighted averages 指数加权平均

在统计学中，叫做exponentially weighted moving average指数加权滑动平均值。

核心思想：
$$
V_t = \beta V_{t-1} + (1-\beta)\theta_t
$$
$\beta$代表的是当前值在最终值中的占比，$\frac{1}{(1-\beta)}$ 代表当前值考虑了之前多少个数据。



### 4. Bias correction in exponentially weighted averages 偏差修正

在指数加权平均的初期，会导致数据偏小，因为缺少之前的数据，可以理解为在热身预测学习。

所用在公式$(13)$的基础上用：
$$
V_t = \frac{V_t}{(1-\beta ^ t)}
$$
来取代公式$(13)$的计算结果，这就叫做偏差修正，当然这么做是基于关注初期数据的需要，如果不关注初期数据可以不采用偏差修正。

### 5. Gradient descent with momentum 动量梯度下降法

- 目的：希望在训练的时候，收敛速度更快同时波动幅度更小

- 思想：

  在某次迭代过程中：用当前的mini-batch来计算当前的dW、db
  $$
  v_{dw} = \beta v_{dw} + (1-\beta) dW
  $$

  $$
  v_{db} = \beta v_{db} + (1-\beta)db
  $$

  $$
  W = W-\alpha v_{dw},b = b-\alpha v_{db}
  $$

  当然也可以有偏差修正，但一般情况采取动量梯度下降不会有偏差的困扰。
  
- 因为小批量梯度下降在只看到一部分示例后进行参数更新，所以更新的方向有一些变化，因此小批量梯度下降所走的路径将“振荡”到收敛。使用动量可以减少这些振荡。
  动量考虑了过去的梯度来平滑更新。先前渐变的“方向”存储在变量中 在形式上，这将是之前步骤中梯度的指数加权平均值。你也可以想到当一个球滚下山时的“速度”，根据山的坡度(坡度的方向)来增加速度（和动量）。

### 6. RMSprop

- 目的：和动量梯度下降法的目的一样，希望收敛速度更快，并且幅度更小

- 思想：

  在某次迭代过程中：用当前的mini-batch来计算当前的dW、db
  $$
  S_{dw} = \beta S_{dw} + (1-\beta) dW^{2}
  $$

  $$
  S_{db} = \beta S_{db} + (1-\beta)db^2
  $$

  $$
  W = W-\alpha \frac{dw}{\sqrt S_{dw}},b = b-\alpha \frac{db}{\sqrt S_{db}}
  $$

  > 本质就是看哪个维度速度慢就让他除以一个小的数让他加快，速度快自然就除以的是一个大数，自然就慢了。



### 7. Adam 优化算法

- 本质：其实将Momentum 和 RMSprop相结合起来。

- 思想：

  在某次迭代过程中：用当前的mini-batch来计算当前的dW、db
  $$
  v_{dw} = \beta_1v_{dw} + (1-\beta_1)dW
  $$
  
  $$
  v_{db} = \beta_1 v_{db} + (1-\beta_1)db
  $$
  
  $$
  S_{dw} = \beta_2 S_{dw} + (1-\beta_2) dW^{2}
  $$

  $$
  S_{db} = \beta_2 S_{db} + (1-\beta_2)db^2
  $$

  $$
  V_{dw}^{corrected} = \frac{V_{dw}}{(1-\beta_1^t)}
  $$

  $$
  V_{db}^{corrected} = \frac{V_{db}}{(1-\beta_1^t)}
  $$

  $$
  S_{dw}^{corrected} = \frac{S_{dw}}{(1-\beta_2^t)}
  $$

  $$
  S_{db}^{corrected} = \frac{S_{db}}{(1-\beta_2^t)}
  $$

  $$
  W = W-\alpha \frac{V_{dw}^{corrected}}{\sqrt {S_{dw}^{corrected}} +  \epsilon },b = b-\alpha \frac{V_{db}^{corrected}}{\sqrt {S_{db}^{corrected}} + \epsilon}
  $$



Adam广泛地使用与各种网络结构，但是其中超参数众多，每一个参数都对应着自己的作用或者说一个moment，所以叫做Adaptive Moment estimation

- $\alpha$需要自行调整
- $\beta_1$一般设置为0.9

- $\beta_2$ 推荐为0.999
- $\epsilon$ 一般设置为$10^{-8}$

### 8. Learning rate decay 学习率衰减

- 目的：神经网络训练是按照mini-batch gradient descent 来进行训练的，所以随着epoch的推移，最终会在最小值的附近的小块区域里摆动，所以无法真正快速的触及到最小值。所以学习率需要随着epoch的增加而衰减，即步子要越跨越小。

  > 模拟退火算法：
  >
  > 

- 思想：

  衰减公式1：
  $$
  \alpha = \frac{1}{1+decay\_rate*epoch\_num}\alpha_0
  $$
  衰减公式2：指数衰减
  $$
  \alpha = 0.95^{epoch\_num}\alpha_0
  $$
  衰减公式3：

  
  $$
  \alpha = \frac{k}{\sqrt{epoch\_num}}\alpha_0
  $$
  





### 9. Code Exercise

- 如果要利用次函数对输入数据X、Y进行随机排序，且要求随机排序后的X Y中的值保持原来的对应关系，可以这样处理：

```python
per = list(np.random.permutation(m))
shuffer_X = X[per]
shuffer_Y = Y[per]
```



## Week3 Hyperparameter Tuning, Batch Normalization and programming Frameworks

### 1. Hyper parameter Tuning 超参数调节（Tips）

- 超参数重要性排序：
  - 学习率$\alpha$
  - 动量梯度下降法中的指数加权平均中的$\beta$，隐层单元数，mini-batch size
  - 层数、学习率衰减公式

- 尝试随机值，因为很难提前知道什么超参数是好的，尝试从粗略到精细，从大区域聚焦到小区域。
- 随机取值并不是在有效值范围内随机均匀取值，而是选择适当的步进值去探究这些超参数，**这里采取==对数标尺==搜索超参数方式会更合理**，反而不用线性轴。因为一般在右边界考虑取值的时候，参数值的细微变化都会引起很大的影响。



- 超参数搜索实践：
  -  Babysitting one model（Panda Approach）：适用于拥有庞大的数据组，但是没有足够的计算机资源，即使正在训练的同时，也可以对参数进修改，就像人类在照看模型一样，根据模型的训练效果进行对参数的调节。在应用领域比如cv 或者推荐系统，一般更多看应用效果，所以采取panda模式。
  - Train many models in parallel（caviar strategy 鱼子酱策略）: 在拥有足够的计算资源的基础上，设置多个不同的参数组，得到不同的模型，然后可以得到不同的学习曲线，从中挑选一个最好的。

### 2. Normalizing activations in a network 对激活函数进行归一化

- 效果：使参数搜索问题变得更容易，使神经网络对参数选择更稳定，对于深度网络更好训练。 

- 核心思想：Batch Normalization Batch 归一化，本质上来讲，其适用的归一化过程不仅仅是输入层，也是对神经网络中的每一个隐藏层的计算结果都进行归一化，因为上一层的结果作为下一层的输入，对输入的归一化可以加速下一层参数的训练效果。

- 过程：

  对于神经网络中的隐藏层的值进行归一化：
  $$
  \mu = \frac{1}{n^{[l]}} \sum_{i=1}^{n^{[l]}}Z_{i}
  $$
  注意这里的$Z_{i}$表示表示的是$Z^{[l]}$的第$i$个分量
  $$
  \sigma^2 = \frac{1}{n^{[l]}}\sum_{i}^{n^{[l]}} (Z_{i}-\mu)^2
  $$

  $$
  Z\_norm_{i} = \frac{Z_{i}-\mu}{\sqrt{\sigma^2+\epsilon}}
  $$

  这样得到的$Z\_norm$就是方差为1，均值为0的分布，但是有的时候隐层的数据拥有不同的分布会存在意义，并不希望其是方差为1均值为0的分布，所以计算$\tilde{Z}$
  $$
  \tilde{Z}_i = \gamma Z\_norm_i + \beta
  $$
  这里的$\gamma$ 和$\beta$是模型训练过程中进行训练的参数，不是超参。

### 3. 深度理解Batch Normalization









