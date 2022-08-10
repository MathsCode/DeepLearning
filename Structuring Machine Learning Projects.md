# Structuring Machine Learning Projects

## Week1 ML strategy

### 1.Orthogonalization 正交化

- 如何在训练集上拟合得更好
  - 更大的网络
  - 更好的优化算法，如Adam
- 如何在交叉验证集上拟合得更好
  - 正则化
  - 增大训练集
- 如何在测试集上拟合得更好
  - 更大的网络
- 如何部署应用的效果更好
  - 换交叉验证集或换损失函数

### 2. Single Number Evaluation Metric 单一数字评估指标

- 准确率Precision: “准确“表示分类器做出相应的决策结果时的准确性，比如当分类器识别为猫时，有准确率这么多可能性真实图片是猫。

- 召回率（查全率）Recall：”查全“表示分类器对于该类物体的所有测试样本，有多少是测试出来的。

- 定义F1 Score，Precision和Recall的调和平均值
  $$
  Score_{F1} = \frac{2}{\frac{1}{P}+\frac{1}{R}}
  $$

### 3. Satisficing and Optimizing Metric 满足和优化指标

选择一个指标作为一个优化指标，其余指标作为满足指标。

### 4. 开发集、测试集的大小设计

### 5. When to Change Dev/Test Sets and Metrics?何时改变开发、测试集和指标

处理机器学习问题时，切分成独立的步骤：

- 弄清楚定义一个什么样的指标来衡量你想做的事情的表现，也就是设定目标（place the target）
- 弄清楚怎么才能在这个指标数优化得更好，也就是怎么对准这个目标精确地射击。



### 6. Why human-level performance？

- 性能最优上限：贝叶斯最优错误率（Bayes optimal error），理论上可以达到的最优错误率

  > - 贝叶斯误差（bayes error rate) 是指在现有特征集上，任意可以基于特征输入进行随机输出的分类器所能达到最小误差。也可以叫做最小误差。
  > - 贝叶斯最优误差是已知真实分布前提下的最优误差
  > - 贝叶斯误差表征了数据力量的极限：数据的力量是有限的，贝叶斯最优误差对应了“**拥有无限真实准确数据时我们能够从数据中汲取出的有效信息的极限**”。事实上，我们利用数据进行预测，就是基于已知数据进行数据分布的预测，而贝叶斯最优误差是在已知分布的前提下进行的，这显然是一种极限状态。

- 如果当前机器学习的算法在某些应用实践上效果比人类的行为要差，那么就可以通过人的操作来对算法进行提升。
  - 让人类来给算法进行标注数据
  - 通过个别的错误案例来判断为什么机器是错误的而人类是对的。
  - 更好分析方差和偏差。

- 对于一个深度学习的模型的效果：Training error和Dev error，并且已知了人类效果

  如果Training error 与人类效果相差很大，说明算法模型对训练集的拟合效果并不好，所以得减少bias，如更大的神经网络，或者更多次的梯度下降

  如果Training error相差不大，但是dev error比较大，关注减少variance。

- **Human-level error as a proxy for Bayes error**，也就是在对模型评判的时候，可以把人类的认知认为是该模型的贝叶斯错误率，并且定义该贝叶斯错误率和训练错误率（Bayes error and Training error）的差值为avoidable bias（可避免误差）



### 7. 目前超过人的表现

- Online advertising
- Product recommendation
- Logistics
- Loan approvals

> 这些都是来源于结构化数据，而对应的是自然感知问题比如CV、NLP、Speech recognition

### 8. Improve your model performance

- 监督学习的两个基本假设
  - 相信算法模型能够对训练集拟合的很好，即avoidable bias 很小
  - 开发集和测试集效果也好，即variance不是很大
- 如果想要提升机器学习系统的性能
  - 了解avoidable bias 有多大，可以知道对模型在训练集上可以优化到多好
    - 更大的网络
    - 更久、更好的优化算法（Momentum、RMSprop、Adam）
    - 神经网络机构、CNN、RNN
  - 了解variance的大小，知道从训练集推广到开发集上还需要做出多少的努力
    - more data
    - 正则化（L2、dropout。。。）

## Week2

### 1. 进行误差分析





