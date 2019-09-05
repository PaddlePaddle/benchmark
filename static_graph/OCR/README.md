# 简介

光学字符识别（Optical Character Recognition，OCR）任务中通常使用的ctc模型和attention模型来识别图片中的文字，这里我们对Paddle的attention模型进行了测试，并且和TensorFlow进行对比。模型来源：
- Paddle测试使用PaddlePaddle/models下面的开源配置[ocr_recognition](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/ocr_recognition)。
- TensorFlow测试使用公司内部OCR团队同事提供的配置。**注意：**该配置为OCR同事用来训练和对齐精度，由于使用的TensorFlow api版本较老，且未对配置进行优化，故这里TensorFlow的测试数据不能代表TensorFlow的最佳性能。

# 准备数据

本测试使用[ocr_recognition](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/ocr_recognition)提供的[开源数据集](http://paddle-ocr-data.bj.bcebos.com/data.tar.gz)，主要包含以下内容：
- 训练数据集，共399425个图像样本
- 训练样本的列表
- 测试数据集，共2000个样本
- 测试样本的列表

注意，Paddle提供的开源数据集中，所有样本图像的大小均相同：高48，宽384。Paddle所使用的列表数据里面存储的是样本的绝对路径，内容如：
```
384 48 53_letter_44164.jpg 75,68,83,83,68,81
384 48 215_UNLOADING_82830.jpg 52,45,43,46,32,35,40,45,38
...
```

tf使用的列表数据里面需要存储样本的绝对路径，内容如：
```
384 48 /benchmark/data/train_images/53_letter_44164.jpg 75,68,83,83,68,81
384 48 /benchmark/data/train_images/215_UNLOADING_82830.jpg 52,45,43,46,32,35,40,45,38
...
```

我们提供了脚本来自动下载数据集，并准备训练和测试的`list`文件。

```
$ git clone https://github.com/PaddlePaddle/benchmark.git 
$ cd benchmark/OCR
$ sh download.sh
... # Downloading and compressing
$ ls
data data.tar.gz download.sh paddle tf
$ cd data
$ ls
test_images test.list test_tf.list train_images train.list train_tf.list
```

# Paddle

请使用`paddlepaddle/paddle:1.3.0-gpu-cuda9.0-cudnn7`以上版本进行测试。

## 训练

我们使用[train.py](paddle/ocr_recognition/train.py)进行训练，需要配置的参数如下：
- `--model`，测试的模型名称，这里我们测试的是`attention`模型
- `--init_model`，预训练的模型参数
- `--train_images`，训练数据集所在目录
- `--train_list`，训练样本的列表
- `--test_images`，测试数据集所在目录
- `--test_list`，测试样本的列表
- `--save_model_dir`，训练所得到的模型的保存目录
- `--use_gpu`，是否使用GPU
- `--parallel`，是否使用多卡训练
- `--batch_size`，训练的batch_size
- `--log_period`，log打印的周期
- `--save_model_period`，保存模型的周期
- `--eval_period`，评测模型的周期
- `--total_step`，训练的轮数

我们将不同的测试项目所需要的配置写在[configs](paddle/configs)目录下不同配置文件中，测试时，你可能需要修改：
- 根据测试机器上的GPU使用情况修改每个配置文件中的`CUDA_VISIBLE_DEVICES`环境变量，例如`export CUDA_VISIBLE_DEVICES=0`。

我们提供统一的脚本`run.sh`下载预训练的参数并解压到`paddle`目录下。但如果你使用的是`paddlepaddle/paddle:1.3.0-gpu-cuda9.0-cudnn7`镜像，由于镜像内没有预装unzip，你可能会碰到解压失败的问题，你可以：
- 预先使用`apt-get install unzip`安装`unzip`工具
- 手动下载[预训练参数](https://paddle-ocr-models.bj.bcebos.com/ocr_attention.zip)并解压到`paddle`目录下

| 测试项目 | 测试命令 |
|---|---|
|GPU单卡训练速度 | `./run.sh train train_gpu_perf_1 > outputs/res_train_gpu_perf_1.txt 2>&1` |
|GPU单卡训练显存占用 | `./run.sh train train_gpu_memory_1`，使用`nvidia-smi`命令观察显存占用 |
|GPU 8卡训练速度 | `./run.sh train train_gpu_perf_8 > outputs/res_train_gpu_perf_8.txt 2>&1` |
|GPU 8卡训练显存占用 | `./run.sh train train_gpu_memory_8`，使用`nvidia-smi`命令观察内存占用 |

## 预测

我们使用[infer.py](paddle/ocr_recognition/infer.py)进行预测，需要配置的参数如下：
- `--model`，测试的模型名称，这里我们测试的是`attention`模型                          
- `--model_path`，预测所使用的模型参数的路径
- `--input_images_list`，预测样本的列表
- `--input_images_dir`，预测所使用的数据集
- `--use_gpu`，是否使用GPU
- `--batch_size`，测试的batch_size
- `--iterations`，测试的轮数
- `--skip_batch_num`，计时所跳过的batch数

我们将不同的测试项目所需要的配置写在[configs](paddle/configs)目录下不同配置文件中，测试时，你可能需要修改：
- 根据测试机器上的GPU使用情况修改每个配置文件中的`CUDA_VISIBLE_DEVICES`环境变量，例如`export CUDA_VISIBLE_DEVICES=0`。
- 预测所使用的模型路径`OCR_model_path`，默认设置为Paddle官方为训练所提供的预训练参数`export OCR_model_path=ocr_attention/ocr_attention_params`。

| 测试项目 | 使用命令 |
|---|---|
|GPU预测速度batch=1 | `./run.sh infer infer_gpu_perf > outputs/res_infer_gpu_perf.txt 2>&1` |
|GPU预测显存占用batch=1| `./run.sh infer infer_gpu_memory`，使用`nvidia-smi`命令观察显存占用 |
|CPU预测速度batch=1 | `./run.sh infer infer_cpu > outputs/res_infer_cpu_perf.txt 2>&1` |
|CPU预测内存占用batch=1 | `./run.sh infer infer_cpu`，使用`top`命令观察内存占用 |

# TensorFlow
## 训练
## 预测