## README

### 运行

`python dataset.py`， 完成数据的预处理和切分

- 由于使用了多线程，不保序，建议直接使用提供的预处理好的文件`data/ori_data_w_sender_only_suffix.json`
- 默认以9:1切分

`python train.py`,  完成训练和测试

- 默认测试sample_rate=1的情况

### 代码简介

#### 数据预处理

`dataset.py`中实现了Email类, 主要方法如下

- preprocess(input_path, output_path)： 数据预处理
- split(path, ratio): 按比例切分训练集、测试集
- split_kfold(path, k)：按k切分得到交叉验证的训练集，测试集

#### 训练, 测试

`train.py`中实现了Trainer类，主要方法如下

- run(train_path, test_path, save_path, k, sample_rate): 完成训练和测试

  > 通过调用`test`, `fit`实现

  - train_path: 训练集路径

  - test_path: 测试集路径

  - save_path: 训练得到的参数保存的路径

  - k: kfold

  - sample_rate: 采样获得不同的训练集大小

- test(test_path, param_path): 加载训练得到的参数文件，完成测试

  - test_path: 测试集路径
  - param_path: 保存的参数文件
  
- fit(train_path, save_path, sample_rate): 加载训练集，统计得到概率，保存到save_path中

  - train_path: 训练集路径
  - save_path: 参数保存路径
  - sample_rate: 采样比例， 得到不同的训练集大小
