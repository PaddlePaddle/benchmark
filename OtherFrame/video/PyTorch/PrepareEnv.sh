#!/usr/bin/env bash

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/PyTorch/)
echo "*******prepare benchmark***********"

################################# 创建一些log目录
export BENCHMARK_ROOT=/workspace # 对应实际地址 benchmark/OtherFrameworks/video/PyTorch/
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.8.0
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/ # benchmark/OtherFrameworks/video/PyTorch/logs/pytorch1.8.0_data_10.2/

if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
# this for update the log_path coding mat
export TRAIN_LOG_DIR=${save_log_dir}/train_log # benchmark/OtherFrameworks/video/PyTorch/logs/pytorch1.8.0_data_10.2/train_log
mkdir -p ${TRAIN_LOG_DIR}

log_path=${TRAIN_LOG_DIR}

################################# 配置python
rm -rf run_env # 
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
# export PATH=/workspace/run_env:${PATH}
export PATH=PATH=${BENCHMARK_ROOT}/run_env:${PATH}

################################# 安装框架
python3.7 -m pip install pip==21.1.1
echo `python3.7 -m pip --version`
python3.7 -m pip install numpy>=1.19 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install torch==1.8.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python3.7 -m pip install torchvision==0.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python3.7 -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
python3.7 -m pip install jupyterlab  -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install pandas>=1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install scikit-learn>=0.22 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install opencv-python>=4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install pyyaml>=5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install yacs>=0.1.6 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install einops>=0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install simplejson -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install av -i https://pypi.tuna.tsinghua.edu.cn/simple/


# git clone https://github.com/HydrogenSulfate/TimeSformer.git # 克隆修改版的竞品repo到本地 benchmark/OtherFrameworks/PyTorch/

# cd TimeSformer # 进入目录

# git checkout dev_benchmark # 切换到修改版的benchmark分支

# python3.7 setup.py build develop # 以包的形式安装

# 根据主项目的配置信息，拉取更新子模块中的代码。
git submodule init
git submodule update

cd ./models/TimeSformer # 进入./models/TimeSformer目录
python3.7 setup.py build develop # 以包的形式安装
cd -


echo "*******prepare benchmark end***********"



