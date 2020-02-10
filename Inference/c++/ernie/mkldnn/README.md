# ERNIE 模型 QAT INT8 精度与性能复现

## 安装与编译PaddlePaddle预测库

- 从Paddle源码编译Paddle推理库，请参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id15)文档。建议编译选项如下：

```bash
PADDLE_ROOT=/path/of/capi
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_PYTHON=ON \
      -DWITH_MKL=ON \
      -DWITH_MKLDNN=ON \
      -DWITH_GPU=OFF  \
      -DON_INFER=ON \
      -DWITH_INFERENCE_API_TEST=ON \
      -DWITH_TESTING=ON \
      ..
 make
 make inference_lib_dist
 make install 
```

## 安装与编译C++性能测试库

- 编译支持MKLDNN的测试程序

```bash
git clone https://github.com/PaddlePaddle/benchmark.git
$ cd benchmark/Inference/c++/ernie/
$ mkdir build
$ cd build
$ cmake -DUSE_GPU=OFF -DPADDLE_ROOT=$PADDLE_ROOT ..
$ make
```

## 精度和性能测试

### 下载模型和数据
* 下载 Ernie 模型
```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/ernie_qat.tar.gz
tar -xzvf ernie_qat.tar.gz
```
解压后模型所在位置： `/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float/`。

* 下载 NLP 数据
```bash
mkdir -p /PATH/TO/DOWNLOAD/NLP/DATASET
cd /PATH/TO/DOWNLOAD/NLP/DATASET/
wget http://paddle-inference-dist.bj.bcebos.com/int8/Ernie_dataset.tar.gz
tar -xzvf Ernie_dataset.tar.gz
```
解压后数据所在位置 `/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset`。

### 精度和性能复现代码
* 精度复现

```bash
model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
dataset_dir=/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_nlp_comparison.py --qat_model=${model_dir} --infer_data=${dataset_dir}/1.8w.bs1 --labels=${dataset_dir}/label.xnli.dev --batch_size=50  --acc_diff_threshold=0.01
```

* 性能复现

```
# 1. 使用PaddlePaddle预测库保存QAT INT8 模型

model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
save_int8_model_path=/PATH/TO/SAVE/INT8/ERNIE/MODEL
save_fp32_model_path=/PATH/TO/SAVE/FP32/ERNIE/MODEL
cd /PATH/TO/PADDLE/build
python ../python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=${model_dir} --int8_model_save_path=${save_int8_model_path} --quantized_ops="fc,reshape2,transpose2"

# 2. 使用benchmark测试库的run.sh 

# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/SAVE/INT8/ERNIE/MODEL`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset/1.8w.bs1`
# uncomment for CPU, use 1 core:
 ./run.sh
# uncomment for CPU, use 20 cores:
 ./run.sh -1 20
```

## 复现结果参考

>**I. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 的精度结果**

|     Model    |  FP32 Accuracy | INT8 QAT Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |      0.80              |         0.82           |     +0.02 |               


>**II. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 单核上单样本耗时**

|     Model    | FP32 Latency (ms) | INT8 QAT Latency (ms)    | Latency Diff |
|:------------:|:----------------------:|:-------------------:|:---------:|
| Ernie        |        318.85          |            95.70    |     3.33   |


>**III. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 20个核上单样本耗时**

|     Model    | FP32 Latency (ms) | INT8 QAT Latency (ms) | Latency Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
| Ernie        |       109.60           |            22.56       |     4.85   |

