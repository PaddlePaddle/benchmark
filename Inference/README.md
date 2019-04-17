#**PaddlePaddle inference benchmarks**
---
inference_benchmark contains implementations of several popular  models inference,and is designed to be as fast as possible. inference_benchmark supports both running on CPU and GPU. It support three mode:native、tensorrt subgraph and anakin subgraph.
##Getting Started
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

---
##Benchmark Model

The following  neural networks are tested with both CPU and GPU. You can use pretrained paddlepaddle fluid model or the model trained by youself.

model name|   inference model
--|--|
ResNet50|
ResNet101|
MobileNetV1|
MobileNetV2|
Vgg16|
Vgg19|
GoogleNet|
ShuffleNet|
MobileNet_SSD|
faster-rcnn|
yolo_v3|
deeplab|
