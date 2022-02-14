#!/usr/bin/env bash
# 执行路径在模型库的根目录下

################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
    --remote-path frame_benchmark/pytorch_req/pytorch_191/ \
    --local-path ./  \
    --mode download
ls

pip install torch-1.9.1-cp37-cp37m-manylinux1_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets>=1.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentencepiece!=0.1.92 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip list

#################################
# 运行 train_cmd 能够自动下载训练数据
echo "*******prepare benchmark end***********"
