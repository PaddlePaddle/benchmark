model_item=VideoSwin
bs_item=1
fp_item=fp16
run_process_type=MultiP
run_mode=DP
device_num=N1C8

bash prepare.sh;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_process_type} ${run_mode} ${device_num} 2>&1;
