#!/usr/bin/env bash
# 拉镜像
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7";
docker pull ${ImageName}

# 启动镜像后测试HRNet
run_cmd="
        cd /workspace/models/HRNet-Image-Classification/;
        cp /workspace/scripts/HRNet-Image-Classification_scripts/*.sh ./;
        cp /workspace/scripts/HRNet-Image-Classification_scripts/analysis_log.py ./;
	bash PrepareEnv.sh;
        bash PrepareData.sh;
	CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 HRNet48C;
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 HRNet48C;
	CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 128 fp32 HRNet48C;
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 128 fp32 HRNet48C;
	mv clas* /workspace/;

	# 启动镜像后测试Twins
        cd /workspace/models/Twins;
        cp /workspace/scripts/Twins_scripts/*.sh ./;
        cp /workspace/scripts/Twins_scripts/analysis_log.py ./;
        bash PrepareEnv.sh;
        bash PrepareData.sh;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 176 fp32 500 alt_gvt_base;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 176 fp32 500 alt_gvt_base;
        mv clas* /workspace/;

	# 启动镜像后测试MobileNetV2, MobileNetV3, ShuffleNetV2, SwinTransformer
        cd /workspace/models/mmclassification;
        cp /workspace/scripts/mmclassification_scripts/*.sh ./;
        cp /workspace/scripts/mmclassification_scripts/analysis_log.py ./;
	bash PrepareEnv.sh;
	bash PrepareData.sh;

	# for MobileNetV2
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 544 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 544 fp32 MobileNetV2 configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py;
        mv clas* /workspace/;
	# for ShuffleNetV2
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 256 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 256 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 1536 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 1536 fp32 ShuffleNetV2 configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py;
        mv clas* /workspace/;
	# for SwinTransformer
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 64 fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 64 fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 104 fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 104 fp32 SwinTransformer configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py;
	mv clas* /workspace/;
	# for MobileNetV3_large_x1_0
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 256 fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 256 fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh sp 640 fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mp 640 fp32 MobileNetV3Large1.0 configs/mobilenet_v3/mobilenet_v3_large_imagenet.py;
	mv clas* /workspace/;
        "


# 启动镜像
nvidia-docker run --name test_pytorch -it  \
    --net=host \
    --shm-size=64g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
nvidia-docker stop test_pytorch
nvidia-docker rm test_pytorch

