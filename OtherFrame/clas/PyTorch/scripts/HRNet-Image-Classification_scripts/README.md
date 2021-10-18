# benchmark使用说明

此目录所有shell脚本是为了测试Twins模型的速度指标，如单卡训练速度指标、多卡训练速度指标等。

## 相关脚本说明

一共有3个脚本：

- `PrepareEnv.sh`: 配置环境
- `PrepareData.sh`: 并下载相应的测试数据，配置好数据路径
- `run_benchmark.sh`: 执行所有训练测试的入口脚本

## 使用说明

**注意**：执行目录为repo的根目录,将所有脚本拷贝到repo根目录

### 1.准备环境及数据

```shell
bash PrepareEnv.sh
bash PrepareData.sh
```

### 2.执行所有模型的测试

```shell
#多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh mp 2 fp32 HRNet48C
#单卡
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh sp 2 fp32 HRNet48C
```

## ips计算方法

日志文件的每条日志输出中含有每个batch的`time_cost`信息。计算`ips`时，去掉前5个输出，剩下的所有`time_cost`求平均，然后用`batch_size / time_cost_avarage`得到`ips`. 
