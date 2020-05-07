# ERNIE 模型 QAT INT8 精度与性能复现

## 安装与编译PaddlePaddle预测库

- 从Paddle源码编译Paddle推理库，请参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/build_and_install_lib_cn.html#id15)文档。建议编译选项如下：

```bash
PADDLE_ROOT=/path/of/capi
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout release/1.8
mkdir build
cd build
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
      -DCMAKE_INSTALL_PREFIX=./tmp \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_PYTHON=ON \
      -DWITH_MKL=ON \
      -DWITH_MKLDNN=ON \
      -DWITH_GPU=OFF  \
      -DON_INFER=ON \
      ..
 make -j$(nproc)
 make inference_lib_dist
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
* 下载 Ernie QAT 模型
```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/ernie_qat.tar.gz
tar -xzvf ernie_qat.tar.gz
```
解压后Ernie QAT模型所在位置： `/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float/`。

* 下载Ernie float32模型
```bash
cd /PATH/TO/DOWNLOAD/MODEL/
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/fp32/ernie_fp32_model.tar.gz 
tar -xvf ernie_fp32_model.tar.gz
```
解压后的Ernie Float32模型在位置：`/PATH/TO/DOWNLOAD/MODEL/ernie_fp32_model`.

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
qat_model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
dataset_dir=/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset
fp32_model_dir=/PATH/TO/DOWNLOAD/MODEL/ernie_fp32_model
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat2_int8_nlp_comparison.py --qat_model=${qat_model_dir} --fp32_model=${fp32_model_dir} --infer_data=${dataset_dir}/1.8w.bs1 --labels=${dataset_dir}/label.xnli.dev --batch_size=50 --batch_num=0 --quantized_ops="fc,reshape2,transpose2,matmul" --acc_diff_threshold=0.01 
```

* 性能复现

#### 1. 使用PaddlePaddle预测库保存QAT INT8模型
```bash
qat_model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
save_int8_model_path=/PATH/TO/SAVE/INT8/ERNIE/MODEL
cd /PATH/TO/PADDLE
python python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=${qat_model_dir} --int8_model_save_path=${save_int8_model_path} --quantized_ops="fc,reshape2,transpose2,matmul"
```
#### 2. Ernie Float32 模型性能复现
```bash
cd /PATH/TO/benchmark/Inference/c++/ernie
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1 
# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/DOWNLOAD/MODEL/ernie_fp32_model`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset/1.8w.bs1`
# For 1 thread performance:
./run.sh
# For 20 threads performance:
./run.sh -1 20
```

#### 3. Ernie QAT INT8 模型性能复现
```bash
cd /PATH/TO/benchmark/Inference/c++/ernie
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1 
# In the file run.sh, set `MODEL_DIR` to `/PATH/TO/SAVE/INT8/ERNIE/MODEL`
# In the file run.sh, set `DATA_FILE` to `/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset/1.8w.bs1`
# For 1 thread performance:
./run.sh
# For 20 threads performance:
./run.sh -1 20
```

## 复现结果参考

>**I. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 的精度结果**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |          80.20%        |         79.64%   |     -0.56%      |               


>**II. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271 上单样本耗时**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:-----------------:|
| 1 thread     |        236.72          |             83.20    |      2.85x       |
| 20 threads   |        27.40           |            14.99     |      1.83x       |

