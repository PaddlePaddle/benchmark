# API配置过滤

对同一个Op而言，从模型中收集到的相似配置在去重后依然非常多。因此，需要有一种筛选配置的方法，保证OP在不同参数设置下的性能数据都能收集到，同时去除大部分冗余的配置。

这里提供2种筛选配置的方法：
- [方法1：通过解析运行log进行筛选](#方法1通过解析运行log进行筛选)
- [方法2：通过解析json文件进行筛选](#方法2通过解析json文件进行筛选)

## 方法1：通过解析运行log进行筛选
通过解析运行log筛选测试配置的方式，筛选的结果会比较准确。
- [第一步：在Paddle Repo中为C++ OP添加log](#第一步在paddle-repo中为c-op添加log)
- [第二步：获取所有Json配置对应的OP运行log](#第二步获取所有json配置对应的op运行log)
- [第三步：根据Op运行log，对所有Json配置进行过滤](#第三步根据op运行log对所有json配置进行过滤)
- [过滤结果展示](#过滤结果展示)

### 第一步：在Paddle Repo中为C++ OP添加log

#### 必要性说明
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
#### 添加log的示例介绍
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

### 第二步：获取所有Json配置对应的OP运行log
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

### 第三步：根据Op运行log，对所有Json配置进行过滤

#### 自动化脚本的使用方式

过滤配置以及保存过滤后的结果都由`benchmark/api/json/select_configs.py`脚本自动完成。只用执行下面的示例命令：
```
python select_configs.py \
      --op_name conv \
      --log_file conv2d.log \
      --input_json_file ../tests/examples/conv2d.json \
      --output_json_file ./conv2d.json
      --parse_runtime_log
```
参数说明：

- op_name：指定log中前向OP名称，例如conv、batch_norm
- log_file：指定log文件的路径
- input_json_file：指定待过滤的json文件路径，其中包含了该OP所有备选配置
- output_json_file：指定输出的json文件路径，则过滤后的配置将被保存在该文件中
- parse_runtime_log：指定过滤方法，设置该参数后，将会通过解析log来筛选配置
- input_shape：可选，指定log中输入shape的名称。未指定时，将默认使用`input_shape`去提取输入的shape。
- ignored_params：可选，不建议设置。如果有特殊需要，希望在过滤配置时，忽略log中的某个参数进行过滤，则设置该参数。

关于`--input_shape`的使用：
- 若log中的输入shape名称不是`input_shape`，例如是`x_dims`，则可以指定 `--input_shape x_dims`用于提取输入的shape。
- 该脚本目前只支持log中给出的输入shape为0个、1个和2个的情况，在shape为1个或者2个时，可以通过设置`--input_shape`指定要提取的输入shape的名字，例如2个输入时，使用`--input_shape x_shape y_shape`，那么脚本会结合`x_shape`和`y_shape`去过滤配置，具体处理过程在“脚本的处理逻辑”小节中有详细介绍。
- 对于输入是list的OP：例如concat、sum和fc，由于list中输入个数不是固定的，并不好在过滤时考虑输入shape。因此如果要根据输入shape去筛选配置，建议手动筛选。如果不需要考虑输入shape，在log中可以不打印shape，那么使用该脚本根据其他参数去筛选配置即可。这种情况不需要设置`--input_shape`。
- 其他复杂的OP：有的OP的输入会有多个，shape的大小会和特定含义的参数有关。例如`generate_proposals`，所有输入的shape大小，与N、A、H、W有关。N是批量大小，A是anchor数，H和W是feature map的高度和宽度。这类OP建议忽略输入shape过滤配置。因此log中可以不打印输入shape，那么同样使用该脚本根据其他参数去筛选配置即可，不需要设置`--input_shape`。

关于`--ignored_params`的使用：
- 若在过滤配置时，只想根据log中的部分参数进行过滤，则可以在运行脚本时指定`--ignored_params param_name1 param_name2`，那么过滤时将忽略这些指定的参数，根据其他参数来过滤配置。
- 该参数通常不需要指定，仅在有特殊需要时使用。例如log中打印了输入shape，其名字为`x_shape`，但是过滤配置时并不希望结合输入shape去过滤配置，那么可以设置`--ignored_params x_shape`。

#### 自动化脚本的处理逻辑

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
  - 第二次分组，对每个标识信息对应的config按照输入shape再次分组，按照shape进行分组的规则为：
    - 当log中没有输入shape信息时，直接进入步骤4。
    - 当log中输入shape只有一个时，根据输入的维数、输入大小是否是2的幂对每个配置的输入打上标签，例如`4-D is_power_of_2=F`表示输入是4D的，并且shape的大小不是2的幂。
    - 当输入shape有2个时：
      - 若两个shape相同，按照shape只有1个时的规则进行分组，例如标签为`is_same_shape=T 4-D-4-D is_power_of_2=T`表示2个输入shape相同，并且shape大小是2的幂。
      - 若两个shape不同，按照两个输入的维数进行标记，例如`is_same_shape=F 4-D-3-D`表示2个输入的shape不同，它们分别是3-D和2-D的。

4. 从每组中分别选取测试配置：
 - 若第3步中没有进行第二次分组，就根据第一次分组的结果，从每组中随机抽取一个。
 - 若第3步中进行了第二次分组，对每组中的配置按照输入shape（如果有2个输入，则对维数更大的那个输入的shape）从小到大排序，从中选取第一个、中间的一个、以及最后一个，得到小、中、大3种shape，然后对shape去重，因此每组最多只能选取3个配置。

5. 最终选取的配置，将会被自动保存到指定的输出文件中，用于API测试。

### 过滤结果展示
以下是按照过滤规则对配置进行分组的结果示例。
#### 0输入的Op

- 当log中没有输入shape时，例如fill_constant，由于是0个输入，因此在log中将不会打印输入shape。可以构造以下测试log：
  ```
  param1=0 param2=1 op=op
  param1=0 param2=1 op=op_grad
  param1=0 param2=1 op=op
  param1=0 param2=1 op=op_grad
  param1=0 param2=1 op=op
  param1=0 param2=1 op=op_grad
  param1=1 param2=1 op=op
  param1=1 param2=1 op=op_grad
  ```
  会得到下面的分组和筛选结果：由于和输入shape无关，因此仅根据其他参数设置进行分组，每组中选择任意1个即可。
  ```
  ==============================config_groups==============================
  config 0: param1=0 param2=1, total: 3.
    Select 1 config_ids: [1].
  config 1: param1=1 param2=1, total: 1.
    Select 1 config_ids: [3].
  ```
#### 1输入的Op
- 当log中仅有1个输入shape时，可以参考conv和batch_norm的log，处理后会得到以下分组结果
  - conv的分组结果：config 0~5代表了6种标识信息，所有配置被分为了6个大组。每组再根据shape细分，最终都只得到1组，即shape 0，因为每个组中shape的标记都是`4-D is_power_of_2=F`，意味着输入都是4-D的，并且shape的大小不是2的幂。当配置数不足3个时，其配置id将被全部选择，如下面的config 0，3，4中的每个shape 0组，包含的id数都只有1个。当配置数超过3个时，从每组中选取最多3个配置，如config 1，2，5中的每个shape 0组，包含的id数都超过了3个。

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
  - batch_norm的分组结果：config 0是第一次忽略输入大小进行分组得到的结果。按照输入shape再次进行分组后，该组又被分为了3个小组，shape 0、shape 1和shape 2。最后从每个小组中选取了3种大小的shape。
    ```
    ==============================config_groups==============================
    config 0: compute_format=NCHW use_global_stats=0 data_layout=NHWC is_inplace=0 test_mode=0, total: 64
      shape 0: 2-D is_power_of_2=T, total: 5. Select 3 config_ids: [49, 48, 46]. The shapes are: [1, 256] [1, 2048] [1, 32768]
      shape 1: 4-D is_power_of_2=F, total: 45. Select 3 config_ids: [44, 33, 11]. The shapes are: [1, 7, 7, 512] [1, 33, 33, 1536] [1, 16, 402, 2048]
      shape 2: 4-D is_power_of_2=T, total: 14. Select 3 config_ids: [35, 51, 59]. The shapes are: [1, 1, 1, 256] [1, 32, 32, 128] [1, 256, 256, 32]
    ```
#### 2输入的Op
- 当log中有2个输入shape时，例如elementwise_add，它有x和y这2个输入，因此log中也会对应打印2个shape。构造以下的测试log模拟这种场景：
  ```
  param1=0 param2=1 x_shape=[16L, 256L, 6L, 6L] y_shape=[16L, 256L, 6L, 6L] op=op
  param1=0 param2=1 x_shape=[16L, 256L, 6L, 6L] y_shape=[16L, 256L, 6L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[16L, 64L, 6L, 6L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[16L, 64L, 6L, 6L] y_shape=[256L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[16L, 32L, 6L, 6L] y_shape=[16L, 256L, 6L, 6L] op=op
  param1=0 param2=1 x_shape=[16L, 32L, 6L, 6L] y_shape=[16L, 256L, 6L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[16L, 256L, 6L, 6L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[16L, 256L, 6L, 6L] y_shape=[256L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[256L, 6L, 6L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[256L, 6L, 6L] y_shape=[256L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[256L, 4L, 4L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[256L, 4L, 4L] y_shape=[256L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[256L, 8L, 8L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[256L, 8L, 8L] y_shape=[256L, 6L] op=op_grad
  param1=0 param2=1 x_shape=[256L, 6L, 6L] y_shape=[256L, 6L] op=op
  param1=0 param2=1 x_shape=[256L, 6L, 6L] y_shape=[256L, 6L] op=op_grad
  ```
  会得到下面的分组和筛选结果：config 0是第一次忽略输入大小分组的结果。然后按照shape再次分组，被分为了4个小组，shape 0、shape 1、shape 2和shape 3。每组中根据维数更大的输入，对shape由小到大排序后进行选择，最多选择3个不同配置。在这个例子中shape 0和shape 1这两组，都是x的维数更大，因此参考x_shape进行排序；而shape 3这一组，x和y维数相同，这种情况下也参考x_shape排序。最后从每个shape组中，最多选择3个配置。需要特别说明的是shape 2，它的标记是`is_same_shape=T 4-D-4-D is_power_of_2=F`，表示输入x和y的shape是相同的，并且都是4-D的，shape的大小不是2的幂。这种情况在上一节中的步骤3中有说明，会按照输入shape为1个的场景进行处理。
  ```
  ==============================config_groups==============================
  config 0: param1=0 param2=1, total: 8.
    shape 0: is_same_shape=F 4-D-2-D, total: 2. Select 2 config_ids: [1, 3]. The shapes are: [[16, 64, 6, 6], [256, 6]] [[16, 256, 6, 6], [256, 6]]
    shape 1: is_same_shape=F 3-D-2-D, total: 4. Select 3 config_ids: [5, 7, 6]. The shapes are: [[256, 4, 4], [256, 6]] [[256, 6, 6], [256, 6]] [[256, 8, 8], [256, 6]]
    shape 2: is_same_shape=T 4-D-4-D is_power_of_2=F, total: 1. Select 1 config_ids: [0]. The shapes are: [[16, 256, 6, 6], [16, 256, 6, 6]]
    shape 3: is_same_shape=F 4-D-4-D, total: 1. Select 1 config_ids: [2]. The shapes are: [[16, 32, 6, 6], [16, 256, 6, 6]]
  ```

## 方法2：通过解析json文件进行筛选

直接解析原始json配置，从中筛选出测试配置的方式，在用法上比较简单。使用该方法，只需要直接运行过滤脚本即可。其筛选的原理与方法1相似，不同之处在于参数值是通过读取json文件直接获得的。以conv2d.json文件中的一条配置为例，它是一个字典。
```
   {
        "op": "conv2d",
        "param_info": {
            "act": {
                "type": "string",
                "value": "None"
            },
            "data_format": {
                "type": "string",
                "value": "NCHW"
            },
            "dilation": {
                "type": "int",
                "value": "1"
            },
            "filter_size": {
                "type": "tuple",
                "value": "(7, 1)"
            },
            "groups": {
                "type": "string",
                "value": "None"
            },
            "input": {
                "dtype": "float32",
                "shape": "[-1L, 1L, 512L, 402L]",
                "type": "Variable"
            },
            "num_filters": {
                "type": "int",
                "value": "32"
            },
            "padding": {
                "type": "tuple",
                "value": "(3, 0)"
            },
            "stride": {
                "type": "tuple",
                "value": "(2, 1)"
            },
            "use_cudnn": {
                "type": "bool",
                "value": "True"
            }
        }
    }
```
上面的字典会先被处理为下面这样一个字符串，与方法1中的log信息在形式上是统一的。所以脚本在对API配置进行分组和筛选时的逻辑也与方法1相同，下文不再赘述。
```
filter_size=(7, 1) act=None data_format=NCHW padding=(3, 0) use_cudnn=True groups=None num_filters=32 stride=(2, 1) dilation=1
```

下面介绍如何使用脚本完成配置筛选，主要介绍3种使用场景以及结果展示：
- [单独过滤某个API的json配置](#单独过滤某个API的json配置)
- [过滤某一类API的json配置](#过滤某一类API的json配置)
- [批量处理所有API的json配置](#批量处理所有API的json配置)
- [过滤结果展示](#过滤结果展示)

### 单独过滤某个API的json配置

如果需要单独过滤某个json文件中配置，可以参考下面的命令：
```
python select_configs.py \
      --input_json_file input_json/conv2d.json \
      --output_json_file output_json/conv2d.json \
      --ignored_params act num_filters
```

参数说明：
- input_json_file：指定待过滤的json文件路径，其中包含了该OP所有备选配置
- output_json_file：指定输出的json文件路径，则过滤后的配置将被保存在该文件中
- ignored_params：可选，通常不建议设置。如果设置了，那么过滤配置时，指定的这些参数将被忽略。指定的参数名必须是json文件中`param_info`里已有的。在以下场景中可以设置：某些参数在API测试脚本中没有使用到，或者参数的值不能穷举，或者已知参数的设置对性能无影响，则可以忽略掉。如conv2d，json文件中的`act`参数在API测试脚本中没有使用，`num_filters`参数实际中会有很多可能的取值，如果不忽略，那么将会得到较多冗余的配置。


### 过滤某一类API的json配置

有些API属于同类API，例如cos、floor、square、tanh、sigmoid、sqrt等，它们同属于activation这一类，这类API通常会使用同一个测试脚本，那么它们的json配置也应该合并后，再从中进行筛选。可以参考下面的命令，过滤脚本将自动地将这些同类API的json配置合并为一个集合，这个集合中的所有配置作为备选配置，然后再从中筛选出这类API的测试配置。
```
python select_configs.py \
      --input_json_file ./input_json \
      --output_json_file ./activation.json \
      --similar_api cos floor square tanh sigmoid sqrt
```
参数说明：

similar_api：用于指定同类API的名称，名称需要与各自的json文件名对应。例如cos对应的是cos.json。

### 批量处理所有API的json配置

如果需要批量过滤所有API的json配置，可以使用`benchmark/api/json/run.sh`脚本。请按照以下几点执行：
- 如果有API需要设置`ignored_params`，那么参考下面conv2d的示例，对`run.sh`脚本做适当修改。其中conv2d是与输入json文件名conv2d.json对应的，`act num_filters`是该API在过滤时需要忽略的参数。
```
# Set ignored params
SPECIAL_DICTS=( \
    ["conv2d"]="act num_filters" \
)
```
- 如果有同类API需要合并它们的json配置后，再进行统一的筛选。那么参考下面activation和elementwise的示例，对`run.sh`脚本做适当修改。其中activation和elementwise分别是这两类API的类名，可以自由指定该名称，同时该名称也将作为输出的json文件名；每个类名都分别对应着一批同类API，这些同类API，例如`cos floor square tanh sigmoid sqrt`分别对应着cos.josn、floor.json、square.json等输入json文件。
```
SIMILAR_API=( \
    ["activation"]="cos floor square tanh sigmoid sqrt" \
    ["elementwise"]="elementwise_add elementwise_div elementwise_max elementwise_min elementwise_mul elementwise_sub elementwise_sum" \
)
```
- 在修改好`run.sh`脚本后，可以执行如下示例命令，根据实际情况指定输入和输出目录，其中`input_json`为输入json的目录，`output_json`为输出json的目录。最终，所有API的json配置将被批量处理，并保存在输出目录。
```
./run.sh input_json output_json
```

### 过滤结果展示

#### 单独过滤某个API的json配置

下面展示的是conv2d的过滤结果，可以看到设置了`ignored_params: ['act', 'num_filters']`后依然产生了不少分组。可以检查过滤结果是否合理，手动地去调整`ignored_params`的设置，以避免留下太多冗余配置。
```
ignored_params: ['act', 'num_filters'].
==============================config_groups==============================
config 0: filter_size=1 data_format=NCHW padding=0 use_cudnn=True groups=1 stride=2 dilation=1 , total: 3.
  shape 0: 4-D is_power_of_2=F, total: 3. Select 3 config_ids: [20, 14, 8]. The shapes are: [[16, 1024, 14, 14]] [[16, 512, 28, 28]] [[16, 256, 56, 56]]
config 1: filter_size=1 data_format=NCHW padding=0 use_cudnn=True groups=1 stride=1 dilation=1 , total: 21.
  shape 0: 4-D is_power_of_2=F, total: 21. Select 3 config_ids: [19, 17, 30]. The shapes are: [[16, 512, 7, 7]] [[16, 1024, 14, 14]] [[16, 256, 56, 56]]
config 2: filter_size=5 data_format=NCHW padding=2 use_cudnn=True groups=1 stride=1 dilation=1 , total: 1.
  shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [24]. The shapes are: [[16, 64, 27, 27]]
config 3: filter_size=3 data_format=NCHW padding=1 use_cudnn=True groups=1 stride=1 dilation=1 , total: 7.
  shape 0: 4-D is_power_of_2=F, total: 7. Select 3 config_ids: [22, 16, 2]. The shapes are: [[16, 512, 7, 7]] [[16, 256, 14, 14]] [[16, 64, 56, 56]]
config 4: filter_size=3 data_format=NCHW padding=1 use_cudnn=True groups=1 stride=2 dilation=1 , total: 3.
  shape 0: 4-D is_power_of_2=F, total: 3. Select 3 config_ids: [18, 12, 6]. The shapes are: [[16, 512, 14, 14]] [[16, 256, 28, 28]] [[16, 128, 56, 56]]
config 5: filter_size=3 data_format=NCHW padding=1 use_cudnn=True groups=32 stride=1 dilation=1 , total: 3.
  shape 0: 4-D is_power_of_2=F, total: 3. Select 3 config_ids: [42, 37, 33]. The shapes are: [[16, 1024, 7, 7]] [[16, 512, 14, 14]] [[16, 256, 28, 28]]
config 6: filter_size=3 data_format=NCHW padding=1 use_cudnn=True groups=32 stride=2 dilation=1 , total: 3.
  shape 0: 4-D is_power_of_2=F, total: 3. Select 3 config_ids: [39, 35, 31]. The shapes are: [[16, 1024, 14, 14]] [[16, 512, 28, 28]] [[16, 256, 56, 56]]
config 7: filter_size=7 data_format=NCHW padding=3 use_cudnn=True groups=1 stride=2 dilation=1 , total: 1.
  shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [0]. The shapes are: [[16, 3, 224, 224]]
config 8: filter_size=11 data_format=NCHW padding=2 use_cudnn=True groups=1 stride=4 dilation=1 , total: 1.
  shape 0: 4-D is_power_of_2=F, total: 1. Select 1 config_ids: [23]. The shapes are: [[16, 3, 224, 224]]
```

#### 过滤某一类API的json配置

下面是对`cos floor square tanh sigmoid sqrt`这一批同类API处理的结果，脚本将它们的配置合并为一个集合，作为备选配置，然后从中筛选出测试配置。
```
ignored_params: None.
==============================config_groups==============================
config 0: , total: 68.
  shape 0: 2-D is_power_of_2=T, total: 10. Select 3 config_ids: [19, 27, 16]. The shapes are: [[2, 1024]] [[1024, 512]] [[4096, 1024]]
  shape 1: 4-D is_power_of_2=F, total: 11. Select 3 config_ids: [57, 52, 49]. The shapes are: [[16, 14, 1, 1]] [[16, 14, 32, 1]] [[16, 3, 256, 256]]
  shape 2: 1-D is_power_of_2=F, total: 6. Select 3 config_ids: [11, 32, 5]. The shapes are: [[3]] [[3072]] [[10000]]
  shape 3: 2-D is_power_of_2=F, total: 25. Select 3 config_ids: [38, 2, 20]. The shapes are: [[2, 768]] [[10000, 200]] [[30522, 1024]]
  shape 4: 4-D is_power_of_2=T, total: 8. Select 3 config_ids: [45, 47, 65]. The shapes are: [[16, 512, 8, 8]] [[16, 128, 32, 32]] [[16, 64, 64, 64]]
  shape 5: 1-D is_power_of_2=T, total: 8. Select 3 config_ids: [0, 67, 13]. The shapes are: [[1]] [[16]] [[4096]]
```
