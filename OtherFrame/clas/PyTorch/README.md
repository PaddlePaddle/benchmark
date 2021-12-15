## 目录 

├── README.md       # 运行文档  
├── models          # 提供竞品Mxnet框架的修改后的模型,官方模型请直接在脚本中拉取,统一方向的模型commit应一致,如不一致请单独在模型运行脚本中写明运行的commit  
├── run_Pytorch.sh  # 运行全量竞品框架模型入口 
├── test_clas.sh    # 全量竞品mxnet框架模型运行脚本  
└── scripts         # 提供各个模型复现性能的脚本  
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
### 2.Docker 镜像,如:

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`   # 
- **pytorch 版本**: `1.9.0`  # 竞品版本：最新稳定版本，如需特定版本请备注说明原因  
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 测试步骤
**注意**：由于batch_size比较大，使用脚本准备的测试数据集很容易出现打不出log的问题，建议使用ImageNet完整的数据集测试，此demo只是保证能运行，用demo数据测试的速度参数会受影响
```bash
bash run_Pytorch.sh;     # 创建容器,在该标准环境中测试模型   
```
脚本内容,如:
```bash
root_dir=${ROOT_DIR:-"/workspace"}                          # /path/to/clas
all_path=${all_path}                                        # /path/to/dataset&whls
log_path_index_dir=${LOG_PATH_INDEX_DIR:-$(pwd)}            # /path/to/result
train_log_dir=${TRAIN_LOG_DIR:-$(pwd)}                      # /path/to/logs
run_plat=${RUN_PLAT:-"local"}                               # wheter downloading dataset

run_cmd="cd /workspace;
         bash test_clas.sh"

# 拉镜像
ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82";
docker pull ${ImageName}

# 启动镜像
nvidia-docker run  -i --rm  \
    --name test_pytorch_clas \
    --net=host \
    --cap-add=ALL \
    --shm-size=64g \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e ROOT_DIR=${root_dir} \
    -e LOG_PATH_INDEX_DIR=${log_path_index_dir} \
    -e TRAIN_LOG_DIR=${train_log_dir} \
    -e RUN_PLAT=${run_plat} \
    -e all_path=${all_path} \
    -v $PWD:/workspace \
    -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    ${ImageName}  \
    /bin/bash -c "${run_cmd}"
nvidia-docker stop test_pytorch_clas
nvidia-docker rm test_pytorch_clas

```

## 输出

每个模型case需返回log解析后待入库数据json文件

```bash
{
"log_file": "clas_MobileNetV2_mp_bs64_fp32_8_speed", \   
"model_name": "clas_MobileNetv2_bs32_fp32", \    # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 如:clas_MobileNetv1_bs32_fp32
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
