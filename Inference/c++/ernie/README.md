# ERNIE模型C++推理Benchmark

## 准备Paddle推理库

- 从Paddle源码编译Paddle推理库，请参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id15)文档。

- 从Paddle官网下载发布的[预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id1)。您需要根据需要部署的服务器的硬件配置（是否支持avx、是否使用mkl、CUDA版本、cuDNN版本），来下载对应的版本。

你可以将准备好的预测库重命名为`paddle_inference`，放置在该测试项目下面，也可以在cmake时通过设置`PADDLE_ROOT`来指定Paddle预测库的位置。

## 编译推理测试代码

- 编译支持GPU的测试程序

``` bash
$ mkdir build
$ cd build
$ cmake -DUSE_GPU=ON -DPADDLE_ROOT=xx/xx ..
$ make
```

- 编译不支持GPU的测试程序

``` bash
$ mkdir build
$ cd build
$ cmake -DUSE_GPU=OFF -DPADDLE_ROOT=xx/xx ..
$ make
```
## 运行可执行程序

该测试程序运行时需要配置以下参数：

- `model_dir`，模型所在目录，注意模型参数当前必须是分开保存成多个文件的。无默认值。
- `data`，测试数据文件所在路径。无默认值。
- `repeat`，每个样本重复执行的次数。默认值为1.
- `warmup_steps`，warmup的步数。默认值为0，即没有warmup。
- `print_outputs`，是否打印预测结果。默认值为false。
- `use_gpu`，是否使用GPU。默认值为false。
- `use_analysis`，是否使用Paddle的`AnalysisPredictor`。默认值为false。
- `profile`，由Paddle预测库中提供，可设置用来进行性能分析。默认值为false。

该项目提供了一个运行脚本`run.sh`，修改了其中的`MODEL_DIR`和`DATA_FILE`后，即可执行`./run.sh`进行CPU预测，执行`./run.sh 0`进行GPU预测。

## 性能测试
在V100 GPU上的性能测试结果

| repeat | 单样本耗时(ms) |
| -----  | -----          |
| 5     | 5.13        |
