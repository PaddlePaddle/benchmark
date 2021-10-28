# NGC PyTorch 性能复现
## 目录 
└── wenet                  # 模型名  
    ├── README.md              # 运行文档  
    ├── analysis.py        # log解析脚本,每个框架尽量统一,可参考[paddle的analysis.py](https://github.com/mmglove/benchmark/blob/jp_0907/scripts/analysis.py)  
    ├── conformer_sp_bs16_fp32_1  # 单卡数据 
    ├── conformer_mp_bs16_fp32_8  # 8卡数据  
    ├── PrepareEnv.sh             #  竞品PyTorch运行环境搭建  
    ├── run_benchmark.sh       # 运行脚本（包含性能、收敛性）  
    ├── run_analysis_mp.sh     # 分析8卡的脚本  
    ├── run_analysis_sp.sh     # 分析单卡的脚本  
    ├── log
    │     ├── log_sp.out    # 单卡的结果
    │     └── log_mp.out    # 8卡的结果
    ├── scripts
    │      └── executor.py    # 用于替换原始wenet项目中的executor.py
    ├── models                 # 提供竞品PyTorch框架的修改后的模型,官方模型请直接在脚本中拉取,统一方向的模型commit应一致,如不一致请单独在模型运行脚本中写明运行的commit
    └── run_PyTorch.sh         # 全量竞品PyTorch框架模型运行脚本

## 环境介绍
### 1.物理机环境
- 单机（单卡、8卡）
  - 系统：Ubuntu 16.04.6 LTS
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 96
  - Driver Version: 440.64.00
  - 内存：440 GB
  - CUDA、cudnn Version: cuda10.2-cudnn7
- 多机（32卡） TODO

### 2.Docker 镜像,如:

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7`   
- **PyTorch 版本**: `1.9.1+cu10`  
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 测试步骤
```bash
run_cmd="bash PrepareEnv.sh
         bash run_PyTorch.sh "
```

## 单个模型脚本目录

## 输出

每个模型case需返回log的json文件

```bash
{
"log_file": "conformer_sp_bs16_fp32_1",
"model_name": "Conformer",
"mission_name": "one gpu", 
"direction_id": 1,
"run_mode": "sp",
"index": 1,
"gpu_num": 1,
"FINAL_RESULT": 27.692,     # ips 值
"JOB_FAIL_FLAG": 0,
"log_with_profiler": null,
"profiler_path": null,
"UNIT": "sent./sec"       # 句子数/秒
}
```



