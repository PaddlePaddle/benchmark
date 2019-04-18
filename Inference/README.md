# **PaddlePaddle inference benchmarks**
---

inference_benchmark contains implementations of several popular  models inference,and is designed to be as fast as possible. inference_benchmark supports both running on CPU and GPU. It support three mode:native、tensorrt subgraph and anakin subgraph.

## Getting Started

To run the Resnet50 inference speed with GPU,run

 `python inference_benchmark.py --model resnet --model_dir models/resnet50 --model_filename model --params_filename params --device gpu`
 
 Some important flags are

+ model: Model to use, e.g. resnet, mobilenet,vgg, googlenet, and shufflenet.
+ model_dir: The dir of inference model file, which is train and save in PaddlePaddle.
+ model_filename: The model graph file in model_file. 
+ params_filename: The params file in model_file.
+ device: CPU or GPU to use.
+ use_tensorrt: If to use TensorRT in inference.It should be use in GPU.
+ use_anakin: If to use Anakin graph in inference.
+ model_precision: The inference precesion,int8 or float32.
+ filename: The input file or data.
+ warmup: The wramup times.
+ repeat: The repeat times in inference.

To see the full list of flags, run python inference_benchmark.py --help

To run Resnet50 in TensorRT,run:

`python inference_benchmark.py --model resnet --model_dir models/resnet50 --model_filename model --params_filename params --device gpu --use_tensorrt`

To run Resnet50 in Anakin,run:

`python inference_benchmark.py --model resnet --model_dir models/resnet50 --model_filename model --params_filename params --device gpu --use_anakin`

---

## Benchmark Model

The following  neural networks are tested with both CPU and GPU. You can use pretrained paddlepaddle fluid model or the model trained by youself.

Model name|  Network file |inference model
--|--|--
ResNet50|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py)|[ResNet50_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/ResNet50_inference.tar)
ResNet101|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet.py)|[Resnet101_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/ResNet101_inference.tar)
MobileNetV1|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet.py)|[MobileNetV1_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/MobileNetV1_inference.tar)
MobileNetV2|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/mobilenet_v2.py)|[MobileNetV2_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/MobileNetV2_inference.tar)
Vgg16|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/vgg.py)|[Vgg16_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/VGG16_inference.tar)
Vgg19|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/vgg.py)|[Vgg19_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/VGG19_inference.tar)
GoogleNet|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/googlenet.py)|[GoogleNet_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/GoogleNet_inference.tar)
ShuffleNet|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/shufflenet_v2.py)|[ShuffleNet_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/shufflenet_inference.tar.gz)
MobileNet_SSD|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/mobilenet_ssd.py)|[MobileNetSSD_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/MobileNet_SSD_infer_model.tar.gz)
faster-rcnn|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/rcnn/models/model_builder.py)|[faster-rcnn_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/faster_rcnn.tar)
yolo_v3|[Code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/models/yolov3.py)|[yolov3_inference_model](https://paddlepaddle-inference-banchmark.bj.bcebos.com/yolo_v3_inference.tgz)

