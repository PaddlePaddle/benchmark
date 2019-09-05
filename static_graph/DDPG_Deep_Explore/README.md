> 去掉参数同步模块

### Run
- TF version
```
cd TF_version
sh run.sh
```

- Fluid version
```
cd Fluid_version
sh run.sh
```

### 12.13 测试结果 （跑10次平均时间）
|       | ensemble_num=1 | ensemble_num=12 |
|-------|----------------|-----------------|
| TF    | 0.48           | 4.25            |
| Fluid | 0.49           | 4.84            |

