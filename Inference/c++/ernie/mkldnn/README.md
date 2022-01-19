# ERNIE QAT INT8 精度与性能复现

## 准备PaddlePaddle预测库

- 用户可以从Paddle源码编译Paddle推理库，请参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)文档。编译选项如下：

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout tags/v2.0.2 -b v2.0.2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_GPU=OFF \
      -DWITH_AVX=ON \
      -DWITH_DISTRIBUTE=OFF \
      -DWITH_MKLDNN=ON \
      -DON_INFER=ON \
      -DWITH_NCCL=OFF \
      -DWITH_PYTHON=ON \
      -DPY_VERSION=3.6 \
      -DWITH_LITE=OFF ..
 make -j$(nproc)
 make inference_lib_dist
 PADDLE_ROOT=path/to/Paddle/build/paddle_inference_install_dir
```


- 用户也可以直接下载 [预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)。请选择 `ubuntu14.04_cpu_avx_mkl` 最新发布版或者develop版。
```
tar -xzvf fluid_inference.tgz
PADDLE_ROOT=full/path/to/fluid_inference
```

## 安装与编译C++性能测试库

- 编译支持MKLDNN的测试程序

```bash
git clone https://github.com/PaddlePaddle/benchmark.git
cd benchmark/Inference/c++/ernie/
mkdir build && cd build
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=$PADDLE_ROOT ..
make
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
quant_model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
dataset_dir=/PATH/TO/DOWNLOAD/NLP/DATASET/Ernie_dataset
fp32_model_dir=/PATH/TO/DOWNLOAD/MODEL/ernie_fp32_model
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/quant2_int8_nlp_comparison.py --quant_model=${quant_model_dir} --fp32_model=${fp32_model_dir} --infer_data=${dataset_dir}/1.8w.bs1 --labels=${dataset_dir}/label.xnli.dev --batch_size=50 --batch_num=0 --ops_to_quantize="fc,reshape2,transpose2,matmul" --acc_diff_threshold=0.01 --targets="fp32,int8"
```

* 性能复现

#### 1. 使用PaddlePaddle预测库保存QAT INT8模型
```bash
quant_model_dir=/PATH/TO/DOWNLOAD/MODEL/Ernie_qat/float
save_int8_model_path=/PATH/TO/SAVE/INT8/ERNIE/MODEL
cd /PATH/TO/PADDLE
# make sure you are under python3.6 (same as the python version Paddle compiled with). Or you could use conda for specific python environment
python python/paddle/fluid/contrib/slim/tests/save_quant_model.py --quant_model_path=${quant_model_dir} --int8_model_save_path=${save_int8_model_path} --ops_to_quantize="fc,reshape2,transpose2,matmul"
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

>**I. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz 的精度结果**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |          80.20%        |         79.44%   |     -0.76%      |


>**II. Ernie QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz 上单样本耗时**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:-----------------:|
| 1 thread     |       228.41          |      75.23           |     3.04X         |
| 20 threads   |       23.93           |      10.98           |     2.18X         |
