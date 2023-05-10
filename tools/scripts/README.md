
## 目录说明
.
GetResourceUtilization.py  # gpu利用率统计脚本
├── analysis.py          # paddle模型通用日志解析脚本
├── README.md
└── run_model.sh         # paddle模型通用脚本

paddle模型执行步骤:
```bash
# clone 解析脚本
git clone https://github.com/PaddlePaddle/benchmark.git -b develop 
export BENCHMARK_ROOT=$PWD/benchmark/tools
# clone 模型库
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop 
# 参考各个模型库文档执行 https://github.com/PaddlePaddle/PaddleClas/blob/develop/test_tipc/docs/benchmark_train.md   
```
