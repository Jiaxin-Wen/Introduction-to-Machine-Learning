## README

温佳鑫 2017010335  wenjx17@mails.tsinghua.edu.cn  

kaggle name: xwwwwwwww

### 数据

放在data/目录下

- 提供了一个预处理脚本`convert.py`，将csv转成json

### 训练

> 以roberta为例

`bash train.sh 0`， 参数是gpu的id

结果会保存到logs/roberta/目录下

### 预测

`VERSION=0 bash predict 0`,  VERSION是训练结果的版本号，参数是gpu的id

会将预测的结果保存到prediction_result目录下

- 保存一个csv文件
- 将probability用json格式保存，为集成做准备

### 融合

prediction_result目录下提供了`ensemble.py`

会根据多个记录模型预测probability的json文件，得到csv文件。







