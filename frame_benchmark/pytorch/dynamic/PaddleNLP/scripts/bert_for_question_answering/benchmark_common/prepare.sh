#!/usr/bin/env bash
#set -xe
echo "*******prepare benchmark***********"

################################# 创建一些log目录,如:
# export BENCHMARK_ROOT=/workspace   # 起容器的时候映射的目录  benchmark/OtherFrameworks/PyTorch/
# github: https://github.com/pytorch/pytorch commit id: e525f433e15de1f16966901604a8c4c662828a8a
run_env=$ROOT_DIR/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`

unset https_proxy && unset http_proxy

wget http://10.181.196.20:8000/downloads/torch_dev_whls.tar
tar -xvf torch_dev_whls.tar
pip install torch_dev_whls/*
pip install transformers psutil astunparse pandas numpy scipy ninja pyyaml setuptools cmake typing_extensions six requests protobuf scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

mv torch torch_tmp
sed -i '1842,1844d' benchmarks/dynamo/common.py
sed -i '929,933d' benchmarks/dynamo/common.py
sed -i '929i \ \ \ \ \ \ \ \ \ \ \ \ os.environ[\"MASTER_ADDR\"] = os.getenv(\"MASTER_ADDR\", \"localhost\")\n            os.environ[\"MASTER_PORT\"] = os.getenv(\"MASTER_PORT\", \"12355\")\n            os.environ[\"RANK\"] = os.getenv(\"RANK\", \"0\")\n            os.environ[\"WORLD_SIZE\"] = os.getenv(\"WORLD_SIZE\", \"1\")\n            torch.cuda.set_device(int(os.environ[\"RANK\"]))\n            torch.distributed.init_process_group(\n                \"nccl\"\n            )' benchmarks/dynamo/common.py

sed -i '330,361d' benchmarks/dynamo/huggingface.py

echo "*******prepare benchmark end***********"
