#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: d9def02050275fa1047cff2763cdbd5a5a51f65d
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.tar' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip install ${dir_name}/*

pip install transformers psutil astunparse pandas numpy scipy ninja pyyaml setuptools cmake typing_extensions six requests protobuf scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

mv torch torch_tmp
sed -i '2410,2414d' benchmarks/dynamo/common.py
sed -i '2410i \ \ \ \ \ \ \ \ \ \ \ \ os.environ[\"MASTER_ADDR\"] = os.getenv(\"MASTER_ADDR\", \"localhost\")\n            os.environ[\"MASTER_PORT\"] = os.getenv(\"MASTER_PORT\", \"12355\")\n            os.environ[\"RANK\"] = os.getenv(\"RANK\", \"0\")\n            os.environ[\"WORLD_SIZE\"] = os.getenv(\"WORLD_SIZE\", \"1\")\n            torch.cuda.set_device(int(os.environ[\"RANK\"]))\n            torch.distributed.init_process_group(\n                \"nccl\"\n            )' benchmarks/dynamo/common.py

sed -i '299,330d' benchmarks/dynamo/huggingface.py

echo "*******prepare benchmark end***********"
