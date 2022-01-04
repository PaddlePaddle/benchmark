# Detection PyTorch 性能复现
## 目录 

```
├── PrepareEnv.sh   # 竞品PyTorch运行环境搭建  
├── README.md       # 说明文档  
├── models          # 提供竞品PyTorch框架的修改后的模型,官方模型请直接在脚本中拉取,统一方向的模型commit应一致,如不一致请单独在模型运行脚本中写明运行的commit  
├── run_PyTorch.sh  # 全量竞品PyTorch框架模型运行脚本  
└── scripts         # 提供各个模型复现性能的脚本  
```

## 环境介绍
### 1.物理机环境
- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - Driver Version: 460.27.04
  - 内存：629 GB
  - CUDA、cudnn Version: cuda10.1-cudnn7 、 cuda11.2-cudnn8-gcc82
- 多机（32卡） TODO
### 2.Docker 镜像:

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`
- **PyTorch 版本**: `1.9.1` 
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 测试步骤
```bash
bash run_PyTorch.sh;     # 创建容器,在该标准环境中测试模型   
```
脚本内容,如:
```bash
#!/usr/bin/env bash

ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/video/PyTorch

run_cmd="cd ${BENCHMARK_ROOT}
        bash PrepareEnv.sh
        cd ${BENCHMARK_ROOT}/models/mmdetection;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 1 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 faster_rcnn;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 fcos;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 deformable_detr;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 deformable_detr;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 8 fp32 1 gfl;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 8 fp32 1 gfl;
        sleep 60;

        cd ${BENCHMARK_ROOT}/models/mmpose;
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        pip uninstall -y mmcv mmcv-full
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.1/index.html
        pip install -r requirements.txt
        pip install -v -e .
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 1 hrnet_w32_keypoint;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 1 hrnet_w32_keypoint;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 160 fp32 1 hrnet_w32_keypoint;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 160 fp32 1 hrnet_w32_keypoint;
        sleep 60;

        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 20 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 20 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 24 fp32 1 higherhrnet_w32;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 24 fp32 1 higherhrnet_w32;
        sleep 60;

        cd ${BENCHMARK_ROOT}/models/SOLO;
        pip uninstall -y mmdet mmcv mmcv-full
        pip install mmcv
        pip install -r requirements.txt
        pip install -v -e .
        cp ${BENCHMARK_ROOT}/scripts/run_benchmark.sh ./;
        cp ${BENCHMARK_ROOT}/scripts/analysis_log.py ./;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 2 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 2 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 4 fp32 1 solov2;
        sleep 60;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 4 fp32 1 solov2;
        sleep 60;
        "

nvidia-docker run --name test_torch_detection -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_detection
nvidia-docker rm test_torch_detection


```
## 输出

每个模型case需返回log解析后待入库数据json文件

```bash
{
"log_file": "/workspace/models/mmdetection/train_log/faster_rcnn_bs1_fp32", \    # log 目录,创建规范见PrepareEnv.sh 
"model_name": "detection_faster_rcnn_bs1_fp32", \    # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 如:clas_MobileNetv1_bs32_fp32
"mission_name": "目标检测", \     # 模型case所属任务名称，具体可参考scripts/config.ini      
"direction_id": 0, \            # 模型case所属方向id,0:CV|1:NLP|2:Rec 具体可参考benchmark/scripts/config.ini    
"run_mode": "sp", \             # 单卡:sp|多卡:mp
"index": 1, \                   # 速度验证默认为1
"gpu_num": 1, \                 # 1|8
"FINAL_RESULT": 7.514, \      # 速度计算后的平均值,需要skip掉不稳定的前几步值
"JOB_FAIL_FLAG": 0, \           # 该模型case运行0:成功|1:失败
}

```



