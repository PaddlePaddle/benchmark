#!/usr/bin/env bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cur_model_list=(dy_seq2seq dy_resnet dy_ptb_lm dy_transformer dy_mobilenet)

# MobileNet
dy_mobilenet(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/mobilenet/
    cd ${cur_model_path}

    # Prepare data
    mkdir -p data
    ln -s ${data_path}/dygraph_data/ILSVRC2012_Pytorch/  ${cur_model_path}/data                         # 准备数据集,需要保证benchmark任务极其21 上对应目录下存在该数据集！

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mobilenet/paddle/run_benchmark.sh ./       # 拷贝脚本到当前目录
    sed -i '/set\ -xe/d' run_benchmark.sh
    modle_list=(MobileNetV2 MobileNetV1)
    for model_name in ${modle_list[@]}
    do
        echo "------------> begin to run ${model_name}"
        echo "index is speed, 1gpu begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 1000 ${model_name} | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 800 ${model_name} | tee ${log_path}/dynamic_${model_name}_speed_8gpus 2>&1
    done
}

# seq2seq
dy_seq2seq(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/seq2seq
    cd ${cur_model_path}

    # Prepare data
    ln -s ${data_path}/dygraph_data/seq2seq/data/ ${cur_model_path}/data

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seq2seq/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 2 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
}

# resnet
dy_resnet(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/resnet
    cd ${cur_model_path}

    # Prepare data
    mkdir -p data
    ln -s ${data_path}/ILSVRC2012  ${cur_model_path}/data                         # 准备数据集,需要保证benchmark任务极其21 上对应目录下存在该数据集！

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 800 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}

# ptb
dy_ptb_lm(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/ptb_lm
    cd ${cur_model_path}

    # Prepare data
    mkdir -p data
    ln -s ${data_path}/dygraph_data/ptb/simple-examples/ ${cur_model_path}/data/

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ptb/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 10000 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
}

# transformer
dy_transformer(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/transformer
    cd ${cur_model_path}

    # Prepare data
    mkdir -p data
    ln -s ${data_path}/dygraph_data/transformer/gen_data/ ${cur_model_path}/

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 3000 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
#    sleep 60
#    echo "index is speed, 8gpus begin"
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 3000 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}
