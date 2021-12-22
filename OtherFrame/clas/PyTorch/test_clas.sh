#!/usr/bin/env bash

# prepare env
rm -rf run_env
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
ln -s $(which python3.7)m-config run_env/python3-config
export PATH=$(pwd)/run_env:${PATH}

apt-get update
apt-get install psmisc -y

echo "----------------------------------------------------- current path is $PWD"
git submodule init
git submodule update

# run HRNet48C models 
cd /workspace/models/HRNet-Image-Classification/;
cp /workspace/scripts/HRNet-Image-Classification_scripts/*.sh ./;
cp /workspace/scripts/HRNet-Image-Classification_scripts/analysis_log.py ./;
bash PrepareEnv.sh;
bash PrepareData.sh;
sed -i '/set\ -xe/d' run_benchmark.sh
bs_list=(64)  # 128)
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run HRNet48C bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp ${bs_item} fp32 HRNet48C;
    sleep 30
    echo "----------begin to run HRNet48C bs${bs_item} mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp ${bs_item} fp32 HRNet48C;
    sleep 30
done
mv clas* ${LOG_PATH_INDEX_DIR};

# run Twins models 
cd /workspace/models/Twins;
cp /workspace/scripts/Twins_scripts/*.sh ./;
cp /workspace/scripts/Twins_scripts/analysis_log.py ./;
bash PrepareEnv.sh;
bash PrepareData.sh;
sed -i '/set\ -xe/d' run_benchmark.sh
bs_list=(64)  #176
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run alt_gvt_base bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp ${bs_item} fp32 500  alt_gvt_base;
    sleep 30
    echo "----------begin to run alt_gvt_base  bs${bs_item} mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp ${bs_item} fp32 500  alt_gvt_base;
    sleep 30
done
mv clas* ${LOG_PATH_INDEX_DIR};

# 启动镜像后测试MobileNetV2, MobileNetV3, ShuffleNetV2, SwinTransformer
cd /workspace/models/mmclassification;
cp /workspace/scripts/mmclassification_scripts/*.sh ./;
cp /workspace/scripts/mmclassification_scripts/analysis_log.py ./;
bash PrepareEnv.sh;
bash PrepareData.sh;
sed -i '/set\ -xe/d' run_benchmark.sh
# for MobileNetV2
bs_list=(64)  #544
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run MobileNetV2 bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
    sleep 30

done
mv clas* ${LOG_PATH_INDEX_DIR};

# for ShuffleNetV2
bs_list=(256)  #1536
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run ShuffleNetV2 bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 256 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
    sleep 30
    echo "----------begin to run ShuffleNetV2 bs${bs_item} mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 256 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
    sleep 30
done
mv clas* ${LOG_PATH_INDEX_DIR};
rm -rf data  # 释放空间

# for SwinTransformer
bs_list=(64)  #104
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run SwinTransformer  bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp ${bs_item} fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
    sleep 30
    echo "----------begin to run SwinTransformer  bs${bs_item} mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp ${bs_item} fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
    sleep 30
done
mv clas* ${LOG_PATH_INDEX_DIR};
rm -rf data  # 释放空间

# for MobileNetV3_large_x1_0
bs_list=(256)  #640
for bs_item in ${bs_list[@]}
do
    echo "----------begin to run MobileNetV3Large1.0 bs${bs_item} sp"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp ${bs_item} fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
    sleep 30
    echo "----------begin to run MobileNetV3Large1.0 bs${bs_item} mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp ${bs_item} fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
    sleep 30
done
mv clas* ${LOG_PATH_INDEX_DIR};
rm -rf data  # 释放空间


