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

- 

