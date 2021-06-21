## Lab3 集成学习报告

> 温佳鑫 2017010335 计84

### 1 实验方法

- 数据集划分：训练集测试集按9：1划分
- 文本向量化：取reviewText字段，用tfidf向量化
- 回归：[bagging, adaboost] X [svm, decision tree] 四种

### 2 实验结果

超参数设置：

- 均使用bootstrap, n=5, sample_rate=0.8
- 决策树的max_depth=20

#### 2.0 不使用集成
| base model    | mae   | rmse  |
| ------------- | ----- | ----- |
| svm           | 0.633 | 0.897 |
| decision tree | 0.757 | 1.058 |


#### 2.1 bagging

| base model    | mae   | rmse  |
| ------------- | ----- | ----- |
| svm           | 0.633 | 0.888 |
| decision tree | 0.733 | 0.991 |

#### 2.2 adaboost


| base model    | mae   | rmse  |
| ------------- | ----- | ----- |
| svm           | 0.643 | 0.871 |
| decision tree | 0.742 | 1.007 |

### 3 分析

根据以上实验结果

- 对比2.0和2.1, bagging与svm或decision tree结合时都有一定效果
- 对比2.0和2.2, adaboost与svm结合时有一定性能损害，与decision tree结合时有一定效果，但相比于bagging的效果要更弱

因此总体来说，在本次实验任务中，bagging的效果要优于adaboost。

上课时我们也曾分析过bagging和boosting的效果，例如”bagging几乎总是有效“，”boosting算法较常出现损害系统性能的情况“，这也与本次实验结果相对应。

关于adaboost的性能为什么会较差，我认为也可以结合课上讲解过的知识点进行分析。

- 弱学习器太弱，导致欠拟合
- boosting可能在有噪声数据上带来性能损失（感觉涉及到打分的label容易有一定程度的噪声）, 而bagging没有这个问题。

### 4 总结

在实验以前我有预想过一些实验结果，比如adaboost将带来非常明显的提升，但与最终的结果有一定差距。这次实验让我更加深刻的认识到课上讲解过的集成学习方法本身的优缺点和适用场景，收获良多。



