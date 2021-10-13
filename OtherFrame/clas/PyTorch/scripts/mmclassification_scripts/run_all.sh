#!/usr/bin/env bash

model_names=("MobileNetV2" "ShuffleNetV2" "SwinTransformer" "MobileNetV3Large")
config_paths=("configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py" "configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py" "configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py" "configs/mobilenet_v3/mobilenet_v3_large_imagenet.py")
batch_sizes=(32 64 16 32)

for i in $(seq 0 `expr ${#model_names[*]} - 1`);do
    _model_name=${model_names[i]}
    _config_path=${config_paths[i]}
    _batch_size=${batch_sizes[i]}

    # for sp
    rm -rf work_dirs
    run_mode=sp
    presion=fp32
    CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh ${run_mode} ${_batch_size} ${presion} ${_model_name} ${_config_path}
    python benchmark/analysis_log.py -d work_dirs -m ${_model_name} -b ${_batch_size} -n 1
    sleep 20

    # for mp
    rm -rf work_dirs
    run_mode=mp
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${_batch_size} ${presion} ${_model_name} ${_config_path}
    python benchmark/analysis_log.py -d work_dirs -m ${_model_name} -b ${_batch_size} -n 8
    sleep 20
done
