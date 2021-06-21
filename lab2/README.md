## README

温佳鑫 2017010335 计84 wenjx17@mails.tsinghua.edu.cn

#### 运行

`python kmeans.py`

- 默认执行：`train(k=10, max_step=300)`

#### 代码简介

`kmeans.py`中实现了Kmeans类，主要方法如下

- load_data()： 读取数据
- fit(): 训练。其中主要调用了三个方法
  - init_centers()：随机初始化中心点
  - update_centers(): 更新中心点
  - assign_clusters(): 更新聚类
- predict(): 按照投票或匹配方法给出预测，并进一步汇报评价指标，给出可视化结果。其中主要调用了两个方法
  - evaluate(): 测指标
  - visualize(): 可视化

#### 训练

基于Kmeans类的成员函数，封装了`train(k, max_step)`, 直接调用即可完成全流程，训练参数自动保存。

#### 测试

基于Kmeans类的成员函数封装了`test(k, load_path)`, 直接调用即可从指定路径加载参数，完成评价指标的汇报和可视化。

