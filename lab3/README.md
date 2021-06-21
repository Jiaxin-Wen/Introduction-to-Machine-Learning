## README

### 运行

#### 预处理

`python preprocess.py` 

会将训练集和测试集保存到data/目录下

#### 训练和测试

`python main.py`

### 代码简介

`main.py`中实现了`AdaBoostRegressor类`和`BaggingRegressor类`

这两类的使用方法类似, 以`BaggingRegressor`为例

```python

# 构造regressor, base_regressor有svm和tree两种
regressor = BaggingRegressor(n=5, ratio=0.8, base_regressor='svm')
# regressor = BaggingRegressor(n=5, ratio=0.8, base_regressor='tree', max_depth=20)


# 训练，会自动将参数(包括模型参数和vectorizer参数)保存到param/目录下
regressor.fit()

# 测试, 可以在fit之后直接调用或单独调用(自动从路径load参数)
regressor.test()

```

