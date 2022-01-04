TensorFlow deeplab开源模型地址，[tensorflow/models/research/deeplab](https://github.com/tensorflow/models/blob/master/research/deeplab)。

### 准备数据

运行benchmark之前，请按照[tensorflow/models/research/deeplab/g3doc/cityscapes.md](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md)准备数据，
并按照[tensorflow/models/research/deeplab/g3doc/model_zoo.md](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
下载相应数据的预训练模型。我们的使用cityscapes数据集进行训练，你也可以使用以下命令直接下载预训练模型：

```
$ wget http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz
```

### 执行测试

请修改`run.sh`，将其中的`TF_MODELS_ROOT`配置成你本地的路径，然后执行`run.sh`进行速度测试。

单卡测试命令：

```
$ CUDA_VISIBLE_DEVICES="0" ./run.sh
```

四卡测试命令：

```
$ CUDA_VISIBLE_DEVICES="0,1,2,3" ./run.sh
```
