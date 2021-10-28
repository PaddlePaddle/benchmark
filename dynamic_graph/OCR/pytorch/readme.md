
# DB.Pytorch benchmark 运行步骤

benchmark目录下的文件用于获取并分析DB.pytorch的训练日志，运行方式如下


## 安装依赖

安装相关依赖和下载数据

```
pip3.7 install -r requirement.txt
bash benchmark/prepare_data.sh
```

## 运行训练

运行分单机八卡运行和单机单卡运行（默认用0号GPU），运行命令如下

```
# 单机单卡
bash benchmark/run_benchmark.sh sp
# 单机8卡
bash benchmark/run_benchmark.sh mp
```

其中，run_benchmark.sh 在执行训练时，会分别训练batch_size=8 和batch_size=16两种情况，所以，全部运行完后，可以得到4个日志文件，如下：

```
pytorch_db_res18_mp_bs16_fp32_8
pytorch_db_res18_mp_bs8_fp32_8
pytorch_db_res18_sp_bs16_fp32_0
pytorch_db_res18_sp_bs8_fp32_0
```

