#!/usr/bin/env bash

## 注意，本脚本仅为示例,相关内容请勿更新到此

# 公共配置文件,配置python 安装pytorch,运行目录:/workspace (起容器的时候映射的目录:benchmark/OtherFrameworks/PyTorch/)
set -xe
echo "*******prepare benchmark***********"
################################# 创建一些log目录,如:
export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
run_env=$BENCHMARK_ROOT/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.8
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
rm -rf $run_env
mkdir $run_env
echo `which python3.7`
ln -s $(which python3.7)m-config  $run_env/python3-config
#ln -s /usr/local/python3.7.0/lib/python3.7m-config /usr/local/bin/python3-config
ln -s $(which python3.7) $run_env/python
ln -s $(which pip3.7) $run_env/pip

export PATH=$run_env:${PATH}

################################# 安装框架 如:
#pip install -U pip
echo `pip --version`
echo `python3-config --help`
pip install torch==1.8.0 -i https://mirror.baidu.com/pypi/simple #https://download.pytorch.org/whl/torch_stable.html
#pip install $BENCHMARK_ROOT/torch-1.8.0-cp37-cp37m-manylinux1_x86_64.whl 
pip install regex pybind11 Ninja -i https://mirror.baidu.com/pypi/simple 
cd $BENCHMARK_ROOT/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd -
##################################  安装其他的公共依赖（单个模型的依赖在中设置,All_PyTorch_Models.sh 中）,如:
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
# dali install
#pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda$(echo ${cuda_version}|cut -d "." -f1)0    # note: dali 版本格式是cuda100 & cuda110

echo "*******prepare benchmark end***********"
