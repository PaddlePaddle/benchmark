
# EAST benchmark测试

benchmark目录下的文件用于获取并分析EAST(tensorflow)的训练日志。训练采用icdar2015数据集，包括1000张训练图像和500张测试图像。模型配置采用resnet50作为backbone，分别训练batch_size=8和batch_size=16的情况。


## 安装依赖

推荐环境：
- CUDA10.1
- CUDNN7.6
- tensorflow-gpu==1.14.0


安装相关依赖和下载数据均在prepare_data.sh 中完成。

```
git clone https://github.com/LDOUBLEV/EAST
cd ./EAST
pip3.7 install -r requirement.txt
bash benchmark/prepare_data.sh
```

注意：EAST需要tensorflow版本1.13-1.15版本，这些版本在CUDA10.1 的环境上无法运行，而Paddle2.0以上版本不再支持CUDA10.1之前的版本，所以为了让tensorflow-gpu在CUDA10.1环境上使用GPU，需要按照如下步骤重定向so文件，其中CUDA10.1不包含cublas.so文件，可以用CUDA10.0的cublas.so文件，下方命令中提供了下载链接：


```
ln -s  /usr/local/cuda-10.1/lib64/libcudart.so.10.1  /usr/local/cuda-10.1/lib64/libcudart.so.10.0
ln -s  /usr/local/cuda-10.1/lib64/libcusparse.so.10  /usr/local/cuda-10.1/lib64/libcusparse.so.10.0
ln -s  /usr/local/cuda-10.1/lib64/libcusolver.so.10  /usr/local/cuda-10.1/lib64/libcusolver.so.10.0
ln -s /usr/local/cuda-10.1/lib64/libcurand.so.10.1.1.243  /usr/local/cuda-10.1/lib64/libcurand.so.10.0
ln -s  /usr/local/cuda-10.1/lib64/libcufft.so.10  /usr/local/cuda-10.1/lib64/libcufft.so.10.0
wget -P /usr/local/cuda-10.1/lib64/  https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/libcublas.so.10
ln -s /usr/local/cuda-10.1/lib64/libcublas.so.10  /usr/local/cuda-10.1/lib64/libcublas.so.10.0
```
注：以上命令只适用于CUDAV10.1.243版本，如果是其他版本，需要重定向的so文件版本号可能不同

安装tensorflow后，执行命令
```
python3.7 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```
如果最终输出为True，则表示安装成功，可以使用GPU训练




## 运行训练benchmark

运行分单机八卡运行和单机单卡运行（默认用0号GPU），运行命令如下

```
# 单机单卡
bash benchmark/run_benchmark.sh sp
# 单机8卡
bash benchmark/run_benchmark.sh mp
```

其中，run_benchmark.sh 在执行训练时，会分别训练batch_size=8 和batch_size=16两种情况，所以，全部运行完后，可以得到4个日志文件，如下：

```
tensorflow_east_resnet50_mp_bs16_fp32_8
tensorflow_east_resnet50_mp_bs8_fp32_8
tensorflow_east_resnet50_sp_bs16_fp32_0
tensorflow_east_resnet50_sp_bs8_fp32_0
```


