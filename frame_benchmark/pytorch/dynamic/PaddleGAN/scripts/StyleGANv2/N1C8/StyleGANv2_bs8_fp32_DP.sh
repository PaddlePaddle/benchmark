model_item=StyleGANv2
bs_item=8
fp_item=fp32
run_mode=DP
device_num=N1C8

sed -i '/set\ -xe/d' run_benchmark.sh
bash prepare.sh ${model_item};
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
