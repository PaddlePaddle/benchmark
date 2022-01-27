# 以下脚本的执行位置在模型套件的目录下
model_item=TimeSformer
bs_item=14
fp_item=fp32
run_process_type=MultiP
run_mode=DP
device_num=N1C8
# get data
bash benchmark/prepare.sh ${model_item} k400;
bash benchmark/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;
