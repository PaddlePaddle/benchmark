# API配置过滤

对同一个Op而言，从模型中收集到的相似配置在去重后依然非常多。因此，需要有一种筛选配置的方法，保证OP在不同参数设置下的性能数据都能收集到，同时去除大部分冗余的配置。
## 为OP添加log
在对OP进行性能优化时，需要分析不同参数配置下的计算开销。对于同一个OP，当API参数设置不同，计算的逻辑也可能不同，那么计算的开销就会有差异。为了准确地获取OP在执行某种参数配置时，计算逻辑落入了哪条分支，就需要为OP添加log。对于一些简单的OP，例如[dist_op](https://github.com/PaddlePaddle/Paddle/blob/7fedf26b8778c5e1da1facfb32bd17ac9ca9f0a0/paddle/fluid/operators/dist_op.h#L99-L121)，影响性能的因素，除了OP输入大小，只有参数`p`。从下面的计算逻辑来看，`p`有4种情况，在忽略输入大小的情况下，最小只需要设计4种不同`p`值的配置，就能保证每个分支的性能都被测试到。
```
if (p == 0) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) != y_t.broadcast(y_bcast_dims))
            .template cast<T>()
            .sum();
  } else if (p == INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .maximum();
  } else if (p == -INFINITY) {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .minimum();
  } else {
    out_t.device(place) =
        (x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims))
            .abs()
            .pow(p)
            .sum()
            .pow(1.0 / p);
  }
```
一些复杂的OP，例如conv_op，计算逻辑中的分支较多，有的分支并不能直接从参数配置中推算出来。例如下面的[is_expand](https://github.com/PaddlePaddle/Paddle/blob/7fedf26b8778c5e1da1facfb32bd17ac9ca9f0a0/paddle/fluid/operators/conv_op.h#L398)分支是根据其他参数计算出的结果，因此在代码中添加log，就能很清楚地了解到OP运行时的计算逻辑落入了哪个分支。
```
bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
...
if (is_expand && data_dim == 2U) {
  col2im(dev_ctx, col, dilations, strides,
         std::vector<int>{paddings[0], paddings[2], paddings[1],
                          paddings[3]}, &in_grad_slice);
} else if (is_expand && data_dim == 3U) {
  col2vol(dev_ctx, col, dilations, strides, paddings, &in_grad_slice);
}
```
添加log的示例，可以参考[conv_op](https://github.com/PaddlePaddle/Paddle/pull/24362)。需要注意：

- 需要将所有类型的kernel都考虑到，例如conv具有GemmKernel，CudnnKernel
- 需要注意为前向、反向的计算都添加log
- 需要将所有影响到分支选择的参数设置打印在一条log中，在log信息的末尾，标明是前向op还是反向op，不同参数之间用空格分离
- log中内容：
  - `op`名，例如`op=conv` 或者`op=conv_grad`
  - 影响到分支选择的参数
  - 影响到数学库算法选择的参数：例如filter_size，在conv op的CudnnKernel计算逻辑中，虽然没有直接影响分支选择，但该参数会影响到cudnn算法的选择。
  - 输入shape
```
    VLOG(10) << " use_cudnn=true"
             << " data_format=" << data_format << " groups=" << groups
             << " is_exhaustive_search=" << exhaustive_search
             << " is_sys_pad=" << is_sys_pad << " input_shape=["
             << input->dims() << "]"
             << " filter_size=[" << filter_dims[2] << ", " << filter_dims[3] << "]"
             << " op=conv";
```

## 获取OP运行log
在过滤之前，首先要获取到OP运行所有配置的log。执行下面的命令将OP运行所有配置时前向、反向的log保存到conv2d.log文件中：
```
GLOG_v=10 python conv2d.py --json_file ./examples/conv2d.json --framework paddle --check_output False --use_gpu=True --backward True --config_id -1 2>&1 | grep 'op=' | awk '{for (i=5;i<=NF;i++){if (i>5) printf(" ");printf("%s", $i)};print ""}'> conv2d.log
```
收集到的log如下，每两条log为一组，代表了同一个配置前、反向的log，例如`op=batch_norm`表示前向log、`op=batch_norm_grad`表示反向log：

- conv
```
use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224] filter_size=[7, 7] op=conv
use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 3, 224, 224] filter_size=[7, 7] op=conv_grad
use_cudnn=true data_format=NHWC groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 56, 56, 64] filter_size=[1, 1] op=conv
use_cudnn=true data_format=NHWC groups=1 is_exhaustive_search=0 is_sys_pad=1 input_shape=[64, 56, 56, 64] filter_size=[1, 1] op=conv_grad
```
- batch_norm
```
data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0 input_shape=[1, 2048] op=batch_norm
data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0 input_shape=[1, 2048] op=batch_norm_grad data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0 input_shape=[1, 32, 32, 128] op=batch_norm
data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0 input_shape=[1, 32, 32, 128] op=batch_norm_grad
```

## 根据log，对配置进行过滤
以下的步骤，都由`benchmark/api/json/select_configs.py`脚本自动完成。只用执行下面的示例命令：
```
python select_configs.py \
      --op_name conv \
      --log_file conv2d.log \
      --input_json_file ../tests/examples/conv2d.json \
      --output_json_file ./conv2d.json
```
参数说明：

- op_name：指定log中前向OP名称，例如conv、batch_norm
- log_file：指定log文件的路径
- input_json_file：指定待过滤的json文件路径，其中包含了该OP所有备选配置
- output_json_file：指定输出的json文件路径，则过滤后的配置将被保存在该文件中
- input_shape：可选，指定log中输入shape的名称。未指定时，将默认使用`input_shape`去提取输入的shape。
- ignored_params：可选，不建议设置。如果有特殊需要，希望在过滤配置时，忽略log中的某个参数进行过滤，则设置该参数。

若log中的输入shape名称不是`input_shape`，例如是`x_dims`，则可以指定 `--input_shape x_dims`用于提取输入的shape。

若在过滤配置时，只想根据log中的部分参数进行过滤，则可以在运行脚本时指定`--ignored_params param_name1 param_name2`，那么过滤时将忽略这些指定的参数，根据其他参数来过滤配置。

脚本的处理逻辑如下：

1. 去除OP名称和输入shape，若设置了`ignored_params`，会同时去除可忽略参数：由`remove_params`函数完成，该函数将从log信息中去除这些参数及其对应的值。这一步是为第一次分组做准备。处理后如下：

- conv前向、反向log：
  ```
  use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7, 7]
  use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7, 7]
  ```
- batch_norm前向、反向log：
  ```
  data_layout=NHWC compute_format=NCHW use_global_stats=0 test_mode=0
  data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0
  ```
2. 根据前向和反向的log，将相同的参数设置和独有的参数设置进行组合，作为分组的参考信息：由`combine_logs_with_key_params`函数完成，该函数实际上是求前向、反向log中参数设置的并集。这样每条配置最终都对应一条标识信息，这条标识信息组合了前向和反向的信息，如下：

- conv前向、反向信息组合后的内容：
  ```
  data_layout=NHWC compute_format=NCHW use_global_stats=0 is_inplace=0 test_mode=0
  ```
- batch_norm前向、反向信息组合后的内容：
  ```
  use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7,7]
  ```
3. 根据每条配置的标识信息和输入大小进行分组，由`grouping_configs`函数完成，分为两步：
  - 第一次分组，根据每条配置的标识信息，将其id划分到相应的组中
  - 第二次分组，对每个标识信息对应的config按照输入shape再次分组。按照shape进行分组的规则有：输入的维数、输入大小是否是2的幂

4. 从每组中分别选取测试配置：按照shape分组后，对每组中的配置按照输入shape从小到大排序，从中选取第一个、中间的一个、以及最后一个，得到小、中、大3种shape。

- conv的分组结果：config 0~5代表了6种标识信息，所有配置被分为了6个大组，每个组中根据shape细分，最终都只得到1组。配置数不足3个时，将被全部选择。否则从每个小组中选取3个配置。

  ```
  ==============================config_groups==============================
  config 0: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[11,11], total: 1
    shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [23]. The shapes are: [64, 3, 224, 224]
  config 1: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[3,3], total: 10
    shape 0: 4-D is_power_of_2=F, total: 10. Select 3 config_ids: [22, 10, 6]. The shapes are: [64, 512, 7, 7] [64, 128, 28, 28] [64, 128, 56, 56]
  config 2: use_cudnn=true data_format=NCHW groups=32 is_exhaustive_search=0 is_sys_pad=1 filter_size=[3,3], total: 6
    shape 0: 4-D is_power_of_2=F, total: 6. Select 3 config_ids: [42, 39, 31]. The shapes are: [64, 1024, 7, 7] [64, 1024, 14, 14] [64, 256, 56, 56]
  config 3: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[5,5], total: 1
    shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [24]. The shapes are: [64, 64, 27, 27]
  config 4: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[7,7], total: 1
    shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [0]. The shapes are: [64, 3, 224, 224]
  config 5: use_cudnn=true data_format=NCHW groups=1 is_exhaustive_search=0 is_sys_pad=1 filter_size=[1,1], total: 24
    shape 0: 4-D is_power_of_2=F, total: 24. Select 3 config_ids: [19, 28, 30]. The shapes are: [64, 512, 7, 7] [64, 64, 56, 56] [64, 256, 56, 56]
  ```
- batch_norm的分组结果：config 0是第一次忽略输入大小进行分组得到的结果。按照输入shape再次进行分组后，该组又被分为了3个小组。最后从每个小组中选取了3种大小的shape。
   ```
   ==============================config_groups==============================
   config 0: compute_format=NCHW use_global_stats=0 data_layout=NHWC is_inplace=0 test_mode=0, total: 64
     shape 0: 2-D is_power_of_2=T, total: 5. Select 3 config_ids: [49, 48, 46]. The shapes are: [1, 256] [1, 2048] [1, 32768]
     shape 1: 4-D is_power_of_2=F, total: 45. Select 3 config_ids: [44, 33, 11]. The shapes are: [1, 7, 7, 512] [1, 33, 33, 1536] [1, 16, 402, 2048]
     shape 2: 4-D is_power_of_2=T, total: 14. Select 3 config_ids: [35, 51, 59]. The shapes are: [1, 1, 1, 256] [1, 32, 32, 128] [1, 256, 256, 32]
   ```

5. 最终选取的配置，将会被自动保存到指定的输出文件中，用于API测试。
