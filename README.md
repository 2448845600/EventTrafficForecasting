# EasyTSF

EasyTSF: **E**xperiment **as**sistant for **y**our **T**ime-**S**eries **F**orecasting

现在已经有 LibCity, BasicTS, Time-Series-Library 等优秀的算法库，我为什么要打造一个新的算法库呢？
- 时序预测的数据集越来越大，模型越来越复杂，现有算法库难以支持简洁的分布式训练和评估；
- 实验过程中，调参是一件低效、枯燥且重复的工作，现有算法库不能简化这一流程。

为此，我们基于 lightning 和 ray.tune，搭建了一套支持实验友好，拓展性强的时序预测算法库，EasyTSF (**E**xperiment **as**sistant for **y**our **T**ime-**S**eries **F**orecasting)

## How to Run?

1. 环境与代码：支持 python >= 3.10, pytorch >= 2.0
```shell
git clone git@github.com:2448845600/easytsf.git
pip install -r requirement.txt
```

2. 准备数据集：直接使用群里的发的压缩包或者百度链接 

链接：https://pan.baidu.com/s/1V33Nfkuw-RJbxZNJQGs_sg?pwd=zb7z 
提取码：zb7z

3. 准备 config 文件
以 'config/reproduce_conf/DLinear/ETTh1.py' 为例：
```python
model_name="DLinear", # 模型名称为DLinear，runner 会加载 easytsf/model/dlinear.py 中的 DLinear 类
dataset_name='ETTh1', # 会从 config/base_conf/datasets.py 找到对应的数据集配置
task="long_time_series_forecasting", # 会使用 config/base_conf/{}.py 对应的任务配置
runner="traffic_forecasting_runner", # 会使用 easytsf/runner/traffic_forecasting_runner.py 对应的 runner
```

4. 运行
一个简单的运行命令：
```shell
python train.py -c config/reproduce_conf/DLinear/ETTh1.py
```

一个简单的超参数搜索命令：
```shell
python ray_tune.py -c config/reproduce_conf/DLinear/ETTh1.py -p config/seed_space.py
```


### 代码结构

```
- easytsf
  - easytsf
    - data
    - layer
    - model
    - util
    - runner
  - config
  - dataset
  - script
  - save
    - [model_name]_[dataset_name]
      - [config_hash]
        - seed_x
          - checkpoints
          - hparams.yaml
          - metric.csv
```
PS:
计算 config_hash 的时候，默认去除 ['seed', 'data_dir', 'save_dir', 'use_wandb']

### 设计思想

多变量时序预测和交通预测非常相似：
- 数据：ETTxx, ECL, Weather, Traffic 和 PEMS0X, METR-LA 的结构都是类似的，可以统一为三个部分：变量(L, N)，时间特征(L, C), 关系矩阵(N, N)。
- 方法：with adj 和 wo adj 两类。
- 评估：时序预测一般使用归一化值计算 mae 和 mse，交通预测一般使用真实值计算 masked mae/rmse/mape
- 设定：时序预测一般是 96 for [48, 96, ..., 720]，交通预测一般是 12 for 12。

为此，我们在 dataloader 中返回归一化后的 variable (L, N) 和 time_marker (L, C)，在 model 中加载 adj_mat，同时在 model 中定义好 inverse_transform 函数

可以考虑将数据集整理为 processed_data:
连续时序数据:
variable.npy : (L, N) # standard norm
variable_scaler.npz : {'mean': (N, ), 'std': (N,)},
time_marker.npy : (L, C) # Year, Month, Day, Hour, Minute, tod, dow, dom, doy 归一化方式 norm_x = TimeNorm(MinMax(x) - 0.5)
adj_mat.npy : (N, N)

非连续时序时序:
variable.npy : (K, l, N) # standard norm
variable_scaler.npz : {'mean': (N, ), 'std': (N,)},
time_marker.npy : (L, C) # Year, Month, Day, Hour, Minute, tod, dow, dom, doy 归一化方式 norm_x = TimeNorm(MinMax(x) - 0.5)
adj_mat.npy : (N, N)
info.txt : split

### 如何实现自己的模型

## 技术框架介绍

### Lightning
Trainer:
 - devices: 
   1. 输入int，表示使用的device的数量，一般代表显卡数量；
   1. 输入list[int]，表示使用的device编号；
   1. 输入str，如果符合 int 格式，比如'1', 按照 int 处理；如果类似 '0,1'，按照 list[int]处理；其他则报错。

   train.py 函数使用字符串输入，'1,'表示使用 1 号 GPU；'1'表示使用一个 GPU，选取GPU的策略由 accelerator 决定; "auto" 表示由程序决定