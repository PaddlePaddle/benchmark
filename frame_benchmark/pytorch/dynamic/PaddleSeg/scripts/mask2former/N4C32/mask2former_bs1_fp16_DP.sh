model_item="mask2former"
bs_item=1
fp_item=fp16
run_mode=DP
device_num=N4C32
max_iter=600
num_workers=8
train_config=configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml

bash prepare.sh;
bash run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_iter} ${num_workers} ${train_config} 2>&1;
