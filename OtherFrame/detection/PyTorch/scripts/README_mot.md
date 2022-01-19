# MOT PyTorch 性能复现
## 目录 

## 单个模型脚本目录
```
└── fairmot                    # 模型名  
    ├── README.md              # 运行文档  
    ├── analysis_log_mot.py    # log解析脚本,每个框架尽量统一,可参考[paddle的analysis.py](https://github.com/mmglove/benchmark/blob/jp_0907/scripts/analysis.py)  
    ├── logs                   # 训练log,注:log中不得包含机器ip等敏感信息  
    │   ├── index              # log解析后待入库数据json文件   
    │   │   ├── modelName_sp_bs32_fp32_1_speed  # 单卡数据  
    │   │   └── modelName_mp_bs32_fp32_8_speed  # 8卡数据  
    │   └── train_log          # 原始训练log  
    ├── prepareMOTData.sh      # 数据处理  
    └── run_benchmark_mot.sh   # 运行脚本（包含性能、收敛性）  
```
## 输出

每个模型case需返回log解析后待入库数据json文件

```bash
{
"log_file": "/workspace/models/fairmot/train_log/fairmot_bs6_fp32", \    # log 目录,创建规范见PrepareEnv_mot.sh 
"model_name": "detection_fairmot_bs6_fp32", \    # 模型case名,创建规范:repoName_模型名_bs${bs_item}_${fp_item} 如:clas_MobileNetv1_bs32_fp32
"mission_name": "目标检测", \     # 模型case所属任务名称，具体可参考scripts/config.ini      
"direction_id": 0, \            # 模型case所属方向id,0:CV|1:NLP|2:Rec 具体可参考benchmark/scripts/config.ini    
"run_mode": "sp", \             # 单卡:sp|多卡:mp
"index": 1, \                   # 速度验证默认为1
"gpu_num": 1, \                 # 1|8
"FINAL_RESULT": 7.514, \      # 速度计算后的平均值,需要skip掉不稳定的前几步值
"JOB_FAIL_FLAG": 0, \           # 该模型case运行0:成功|1:失败
}
```