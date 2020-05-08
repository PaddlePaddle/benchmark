# API配置过滤

## 获取OP运行log
在过滤之前，首先要获取到OP运行所有配置的log：

- 对于大部分OP可以用`GLOG_vmodule`输出log，这样的log冗余的信息比较少。
```
GLOG_vmodule=batch_norm_op=10 python batch_norm.py --json_file ./examples/batch_norm.json --framework paddle --check_output False --use_gpu=True --backward True --config_id -1 2>&1 | grep 'op=' > batch_norm.log
```
- 但对于有的OP，例如conv，存在`conv_op`，`conv_cudnn_op`，输出log时，可以用下面这种方式，确保运行不同kernel时的log都被收集到。
```
GLOG_v=10 python conv2d.py --json_file ./examples/conv2d.json --framework paddle --check_output False --use_gpu=True --backward True --config_id -1 | grep 'op=' > conv2d.log
```
收集到的log如下，每两条log为一组，代表了同一个配置前、反向的log，例如`op=batch_norm`表示前向log、`op=batch_norm_grad`表示反向log：

- batch_norm
```
I0508 09:15:03.065099 22585 batch_norm_op.cu:80] op=batch_norm data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0
I0508 09:15:03.070510 22585 batch_norm_op.cu:576] op=batch_norm_grad data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0
I0508 09:15:03.258034 22585 batch_norm_op.cu:80] op=batch_norm data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0
I0508 09:15:03.263720 22585 batch_norm_op.cu:576] op=batch_norm_grad data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0
```
- conv
```
I0508 08:40:36.164196 22388 conv_cudnn_op.cu:142] op=conv use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224] filter_size=[7, 7]
I0508 08:40:36.179255 22388 conv_cudnn_op.cu:442] op=conv_grad use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224] filter_size=[7, 7]
I0508 08:40:37.576503 22388 conv_cudnn_op.cu:142] op=conv use_cudnn=true data_format=NHWC groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 56, 56, 64] filter_size=[1, 1]
I0508 08:40:37.579442 22388 conv_cudnn_op.cu:442] op=conv_grad use_cudnn=true data_format=NHWC groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 56, 56, 64] filter_size=[1, 1]
```

## 根据log，对配置进行过滤
- 首先进一步对每组前、反向的log信息处理，只保留必要部分，如：
```
op=batch_norm data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0
op=batch_norm_grad data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0
```
```
op=conv use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224]  filter_size=[7, 7]
op=conv_grad use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224] filter_size=[7, 7]
```
- 去除可忽略的参数：对于batch_norm，`op`并不是影响分组的关键信息，对于conv，`op`和`input_shape`并不是影响分组的关键信息，先去掉。则前向和反向的log信息如下：
  - 第一组为batch_norm前向、反向信息
  - 第二组为conv前向、反向信息
```
data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0
data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0
```

```
use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7, 7]
use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7, 7]
```
- 从前向和反向的log中，取关键参数进行组合，作为分组的参考信息：即求前向、反向log中参数设置的并集。这样每条配置最终都对应一条标识信息，这条标识信息组合了前向和反向的信息，如下：
  - 第一条为上面batch_norm前向、反向信息组合后的内容
  - 第二条为上面conv前向、反向信息组合后的内容
```
"data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0 test_mode=0"
```
```
"use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7,7]"
```
- 根据每条配置的标识信息进行分组

- 从每组中分别随机选取测试配置：若未设置`num_configs`，则抽取的参考数目为组数，否则为设置的个数。假设未设置`num_configs`，则下面的例子中，`num_configs=6`。根据每组配置数占总配置数的比例随机抽取，以conv2d为例，json配置一共有43个：：
  - 第一组共24个，需要抽取的个数为：max(1,  6*24/43) = 3
  - 第二组只有1个，需要抽取的个数为：max(1,  6*1/43) = 1
```
ignored_params: ['input_shape'].
==================config_groups===================
config: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[1,1], total: 24
Select 3 config_ids: [38, 8, 29].
config: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[5,5], total: 1
Select 1 config_ids: [24].
config: use_cudnn=true data_format=NCHW groups=32 is_exhaustive_search=0 is_sys_pad=1 filter_size=[3,3], total: 6
Select 1 config_ids: [33].
config: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[11,11], total: 1
Select 1 config_ids: [23].
config: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[3,3], total: 10
Select 1 config_ids: [18].
config: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7,7], total: 1
Select 1 config_ids: [0].
```
