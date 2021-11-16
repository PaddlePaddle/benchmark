# Parakeet PyTorch 性能复现
## 目录 

## 单个模型脚本目录
```text
PWAGN                      # 模型名  
├── README.md              # 运行文档  
├── analysis_log.py        # log解析脚本
├── logs                   # 训练log,注:log中不得包含机器ip等敏感信息  
│   ├── index              # log解析后待入库数据json文件   
│   │   ├── Parakeet_PWGAN_mp_bs6_fp32_2_speed  # 单卡数据  
│   │   └── Parakeet_PWGAN_mp_bs6_fp32_8_speed  # 8卡数据  
│   └── train_log          # 原始训练log  
│       ├── Parakeet_PWGAN_mp_bs6_fp32_2  # 单卡数据
│       └── Parakeet_PWGAN_mp_bs6_fp32_8  # 8卡数据
├── preData.sh             # 数据处理  
└── run_benchmark.sh       # 运行脚本（包含性能、收敛性）  
```

## 输出

每个模型case需返回log解析后待入库数据json文件

```bash
{
"log_file": "/workspace/scripts/logs/train_log/Parakeet_PWGAN_sp_bs6_fp32_1",  # log 目录
"model_name": "Parakeet_PWGAN_bs6_fp32",   # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 如:clas_MobileNetv1_bs32_fp32 
"mission_name": "语音合成",                 # 模型case所属任务名称，具体可参考scripts/config.ini 
"direction_id": 1,                         # 模型case所属方向id,0:CV|1:NLP|2:Rec 具体可参考benchmark/scripts/config.ini
"run_mode": "sp",                          # 单卡:sp|多卡:mp
"index": 1,                                # 速度验证默认为1
"gpu_num": 1, # 1|8                        # 1|8
"FINAL_RESULT": 18.48883604166667,         # 速度计算后的平均值,需要skip掉不稳定的前几步值
"JOB_FAIL_FLAG": 0,                        # 该模型case运行0:成功|1:失败
"UNIT": "sequences/sec"                    # 速度指标的单位 
}

```



