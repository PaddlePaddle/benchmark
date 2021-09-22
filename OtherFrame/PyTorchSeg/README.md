# PyTorch 分割模型 性能复现
## 目录 

├── README.md           # 说明文档 
├── run_PyTorch.sh      # 执行入口，包括环境搭建、测试获取所有分割模型的训练性能 
├── PrepareEnv.sh       # PyTorch和mmsegmentation运行环境搭建、训练数据下载
├── analysis_log.py     # 分析训练的log得到训练性能的数据
└── run_benchmark.sh    # 执行实体，测试单个分割模型的训练性能
## 环境介绍
### 物理机环境
- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - CUDA、cudnn Version: cuda10.2-cudnn7
### Docker 镜像

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`
- **PyTorch 版本**: `1.9.1`  # 竞品版本：最新稳定版本，如需特定版本请备注说明原因  
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

run_cmd="cp /workspace/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /home/mmsegmentation;
         cp /workspace/run_benchmark.sh ./;
         cp /workspace/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ocrnet_hrnetw48 sp fp32 2 500 5;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ocrnet_hrnetw48 mp fp32 2 500 5;
         "

nvidia-docker run --name test_torch_seg -it  \
    --net=host \
    --shm-size=1g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"

nvidia-docker stop test_torch_seg
nvidia-docker rm test_torch_seg
```

## 输出

```bash
{
"log_file": "/logs/2021.0906.211134.post107/train_log/ResNet101_bs32_1_1_sp", \    # log 目录,创建规范见PrepareEnv.sh 
"model_name": "clas_MobileNetv1_bs32_fp32", \    # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 如:clas_MobileNetv1_bs32_fp32
"mission_name": "图像分类", \     # 模型case所属任务名称，具体可参考scripts/config.ini      
"direction_id": 0, \            # 模型case所属方向id,0:CV|1:NLP|2:Rec 具体可参考benchmark/scripts/config.ini    
"run_mode": "sp", \             # 单卡:sp|多卡:mp
"index": 1, \                   # 速度验证默认为1
"gpu_num": 1, \                 # 1|8
"FINAL_RESULT": 197.514, \      # 速度计算后的平均值,需要skip掉不稳定的前几步值
"JOB_FAIL_FLAG": 0, \           # 该模型case运行0:成功|1:失败
"UNIT": "images/s" \            # 速度指标的单位 
}

```



