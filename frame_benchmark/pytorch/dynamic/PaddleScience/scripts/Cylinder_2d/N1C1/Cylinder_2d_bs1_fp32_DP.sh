model_item=Cylinder_2d
bs_item=1
fp_item=fp32
run_mode=DP
device_num=N1C1
export FLAG_TORCH_WHL_URL=https://paddle-wheel.bj.bcebos.com/benchmark/torch_2_0_0_whls.tar
pip config set global.index-url ${FLAG_TORCH_WHL_URL}
pip config list
#prepare
echo -e "FLAG_TORCH_WHL_URL is : " $FLAG_TORCH_WHL_URL "\n"
bash prepare.sh
#run
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} 2>&1;
sleep 10;