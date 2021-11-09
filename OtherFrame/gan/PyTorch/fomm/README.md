# PyTorch 生成模型 性能复现
## 目录 

```
├── README.md               # 说明文档 
├── run_PyTorch.sh          # 执行入口，包括环境搭建、测试获取所有生成模型的训练性能 
├── scripts/PrepareEnv.sh   # PyTorch和MMEditing运行环境搭建、训练数据下载
├── scripts/analysis_log.py         # 分析训练的log得到训练性能的数据
├── scripts/run_benchmark.sh        # 执行实体，测试单个生成模型的训练性能
└── models                          # 提供竞品PyTorch框架的repo
```

## 环境介绍
### 物理机环境
- 单机（单卡、4卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - CUDA、cudnn Version: cuda10.2-cudnn7

### Docker 镜像

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`
- **PyTorch 版本**: `1.0.0` 
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 测试步骤

```bash
bash run_PyTorch.sh;     # 创建容器,在该标准环境中测试模型   
```

如果在docker内部按住torch等框架耗时很久，可以设置代理。下载测试数据的时候，需要关闭代理，否则下载耗时很久。

脚本内容,如:

```bash
#!/usr/bin/env bash
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}
run_cmd="cd /workspace;
         cp /workspace/scripts/PrepareEnv.sh ./;
         bash PrepareEnv.sh;
         cd /workspace/first-order-model/;
         cp /workspace/scripts/run_benchmark.sh ./;
         cp /workspace/scripts/analysis_log.py ./;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh fomm_sp_bs8 sp fp32 8 300 4;
         CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh fomm_sp_bs16 sp fp32 16 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh fomm_mp_bs32 mp fp32 8 300 4;
         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh fomm_mp_bs64 mp fp32 16 300 4;
         "
         
nvidia-docker run --name test_torch_gan -it  \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
nvidia-docker stop test_torch_gan
nvidia-docker rm test_torch_gan
```

## 输出

执行完成后，在当前目录会产出分割模型训练性能数据的文件，比如`fomm_sp_bs8_fp32_1_speed`等文件，内容如下所示。

```bash
{
"log_file": "/workspace/models/fomm/fomm_sp_bs8_fp32_1", \    # log 目录,创建规范见PrepareEnv.sh 
"model_name": "fomm_sp_bs8", \    # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 
"mission_name": "图像生成", \         # 模型case所属任务名称，具体可参考scripts/config.ini      
"direction_id": 0, \                 # 模型case所属方向id,0:CV|1:NLP|2:Rec 具体可参考benchmark/scripts/config.ini    
"run_mode": "sp", \                  # 单卡:sp|多卡:mp
"index": 1, \                        # 速度验证默认为1
"gpu_num": 1, \                      # 1|8
"FINAL_RESULT": 75.655, \            # 速度计算后的平均值,需要skip掉不稳定的前几步值
"JOB_FAIL_FLAG": 0, \                # 该模型case运行0:成功|1:失败
"UNIT": "images/s" \                 # 速度指标的单位 
}
```

