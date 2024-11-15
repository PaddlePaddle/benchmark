#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

# wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
# tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
# dir_name=$(echo ${tar_file_name} | awk -F '.tar' '{print $1}')
# tar xf ${tar_file_name}
# rm -rf ${tar_file_name}

RUN_SETUP=${RUN_SETUP:-"true"}
if [ "$RUN_SETUP" = "true" ]; then
    # pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
    # pip install ${dir_name}/*
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
    # pip install --upgrade tokenizers
    # pip install --upgrade transformers
    # # pip install pybind11>=2.12
    # pip install install accelerate==0.29.3  pandas numpy scipy datasets diffusers==0.29.0 ftfy safetensors tensorboard
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
# rm -rf stable-diffusion-3-medium-diffusers
# rm -rf stable-diffusion-3-medium-diffusers.tar
# rm -rf dog
# rm -rf dog.zip
# wget https://paddlenlp.bj.bcebos.com/models/community/westfish/sd3/stable-diffusion-3-medium-diffusers.tar
# tar -xvf stable-diffusion-3-medium-diffusers.tar
# wget https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip
# unzip dog.zip


# 解决compile下报错的问题
# ln -s /usr/include/python3.10 /usr/local/include/ 
echo "*******prepare benchmark end***********"
