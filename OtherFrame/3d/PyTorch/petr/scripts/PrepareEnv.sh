#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/gan/PyTorch/mmedting)
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.9.0
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/

if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
# this for update the log_path coding mat
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}

log_path=${TRAIN_LOG_DIR}

################################# 配置python, 如:
rm -rf run_env
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
export PATH=/workspace/run_env:${PATH}

################################# 安装框架 如:
pip install -U pip
echo `pip --version`
pip install torch==1.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

cd /workspace/models
git clone https://github.com/open-mmlab/mmdetection.git
cd /workspace/models/mmdetection
git checkout v2.24.1 
pip install -r requirements/build.txt
python setup.py develop

pip install mmsegmentation==0.20.2

cd /workspace/models
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd /workspace/models/mmdetection3d
git checkout v0.17.1 
pip install -r requirements/build.txt
python setup.py develop

cd /workspace/models/petr
mkdir ckpts
ln -s /workspace/models/mmdetection3d /workspace/models/petr/mmdetection3d

################################# 准备训练数据 如:
mkdir -p data
# 由于nuscenes数据集太大，为避免每次下载过于耗时，请将nuscenes数据集拷贝到data目录下
# 并软链到/data/Dataset/nuScenes目录
# cp -r /nuscenes_dataste_root data/
# ln -s /nuscenes_dataste_root /data/Dataset/nuScenes

echo "*******prepare benchmark end***********"
