# PaddleVideo 下benchmark模型执行说明
通用检测benchmark测试脚本说明
# benchmark的文件夹结构
```bash
benchmark
├── README.md
├── configs
│   ├── timesformer.yaml
│
├── prepare.sh    # 相关数据准备脚本，完成数据、模型的自动下载
├── run_all.sh    # 批量模型执行脚本
└── run_benchmark.sh    # 单模型运行脚本，可完成指定模型的测试方案
```

# Docker 运行环境
docker image: paddlepaddle/paddle:latest-gpu-cuda11.2-cudnn8
paddle = 2.2.2
python = 3.7

# 运行benchmark测试
```bash
git clone https://github.com/PaddlePaddle/PaddleVideo.git # clone官方repo
cd PaddleVideo # 进入repo
bash benchmark/prepare.sh k400 # 下载并准备好小数据集(以k400为例)
```

# 运行指定模型
```bash
bash run_benchmark.sh ${model_item} {batch_size} ${fp_item} {run_mode} ${device_num}
model_item: TimeSformer # (以TimeSformer为例)
```

# 单卡
```bash
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh TimeSformer 1 fp16 SingleP DP N1C1 # (以TimeSformer为例)
```

# 多卡
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh TimeSformer 1 fp16 MultiP DP N1C8 # 8卡(以TimeSformer为例)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmark/run_benchmark.sh TimeSformer 1 fp16 MultiP DP N1C4 # 4卡(以TimeSformer为例)
```
# 打开Profiling开关运行，只用跑单卡
```bash
export PROFILING=true
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh TimeSformer 1 fp16 SingleP DP N1C1 # (以TimeSformer为例)
```

# profiling 目前只跑这一种默认的配置
 add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
