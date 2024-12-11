#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy


RUN_SETUP=${RUN_SETUP:-"true"}
if [ "$RUN_SETUP" = "true" ]; then
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 
    # --index-url https://download.pytorch.org/whl/cu118
    pip install diffusers==0.29.0 
    pip install sentencepiece==0.2.0
    pip install accelerate==1.1.1
    pip install transformers==4.46.2
    pip install tensorboard==2.18.0
    pip install protobuf==5.28.3
    pip install datasets==3.1.0
    pip install peft==0.13.2
else
    echo "Skipping setup and installation steps as RUN_SETUP is set to false."
fi

if [ ! -d "stable-diffusion-3-medium-diffusers" ]; then
    rm -rf stable-diffusion-3-medium-diffusers.tar
    echo "Downloading stable-diffusion-3-medium-diffusers.tar..."
    wget https://paddlenlp.bj.bcebos.com/models/community/westfish/sd3_benchmark/stable-diffusion-3-medium-diffusers.tar
    echo "Extracting stable-diffusion-3-medium-diffusers.tar..."
    tar -xvf stable-diffusion-3-medium-diffusers.tar
else
    echo "Directory stable-diffusion-3-medium-diffusers already exists. Skipping download."
fi

if [ ! -d "dog" ]; then
    rm -rf dog.zip
    echo "Downloading dog.zip..."
    wget https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip
    echo "Unzipping dog.zip..."
    unzip dog.zip
else
    echo "Directory dog already exists. Skipping download."
fi

echo "*******prepare benchmark end***********"
