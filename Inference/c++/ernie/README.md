# ERNIE模型C++ inference


## 数据预处理
首先需要对数据进行处理，在python的训练、测试数据的基础上再进行一些处理，包括 tokenization，batching，numericalization，并且把处理后的数据输出为文本文件。使用方法如下：
```
sh run_gen.sh
```

**生成的数据格式**

生成的数据一行代表一个 `batch`, 包含五个字段

```text
src_ids, pos_ids, sent_ids, self_attn_mask, labels, index_list, next_sent_index
```

字段之间按照分号(;)分隔，其中各字段内部 `shape` 和 `data` 按照冒号(:)分隔，`shape` 和 `data` 内部按空格分隔，`self_attention_bias` 为 FLOAT32 类型，其余字段为 INT64 类型。


## 模型和配置
首先需要下载inference model
```
hadoop fs -get hdfs://yq01-global-hdfs.dmop.baidu.com:54310/user/ccdb/working/nlp/ol/caoyuhui/duxiaoman/inference_c_lib/256_inference_model.tar.gz .
tar -xvf 256_inference_model.tar.gz
hadoop fs -get hdfs://yq01-global-hdfs.dmop.baidu.com:54310/user/ccdb/working/nlp/ol/caoyuhui/duxiaoman/inference_c_lib/512_inference_model.tar.gz .
tar -xvf 512_inference_model.tar.gz
```

然后，需要配置配置文件，包括：gen_data.config、infer_gpu.config、infer_cpu.config


## 编译和运行

**已有编译好的Fluid_inference，在当前目录运行如下命令即可：（后续的下载过程可忽略**
```
hadoop fs -get hdfs://yq01-global-hdfs.dmop.baidu.com:54310/user/ccdb/working/nlp/ol/caoyuhui/duxiaoman/inference_c_lib/fluid_inference.tar.gz .
tar -xvf fluid_inference.tar.gz
```

为了编译 inference demo，`c++` 编译器需要支持 `C++11` 标准。

首先下载对应的 [Fluid_inference库](http://paddlepaddle.org/documentation/docs/zh/1.2/advanced_usage/deploy/inference/build_and_install_lib_cn.html) , 根据使用的 paddle 的版本和配置状况 (是否使用 avx, mkl, 以及 cuda, cudnn 版本) 选择下载对应的版本，并解压至 `inference` 目录，会得到 `fluid_inference` 文件夹。


**已有编译好的build，在当前目录下运行如下命令即可：（后续的编译过程可以忽略）**
```
hadoop fs -get hdfs://yq01-global-hdfs.dmop.baidu.com:54310/user/ccdb/working/nlp/ol/caoyuhui/duxiaoman/inference_c_lib/build.tar.gz .
tar -xvf build.tar.gz
```

如果改动inference.cc，则需要用以下命令重新编译：
``` bash
mkdir build
cd build
cmake
make
```

**运行 inference**
使用GPU：
```
sh run_infer_gpu.sh &>run_log &
```
使用CPU
```
sh run_infer_cpu.sh &>run_log &
```

## 性能测试
在V100 GPU上的性能测试结果

| repeat | 单样本耗时(ms) |
| -----  | -----          |
| 5     | 5.13        |
