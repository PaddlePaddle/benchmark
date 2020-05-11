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

cur_model_list=(dy_mobilenet)

#run MobileNet
dy_mobilenet(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/mobilenet/
    cd ${cur_model_path}

    # Prepare data
    mkdir -p data
    ln -s ${data_path}/ILSVRC2012  ${cur_model_path}/data
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mobilenet/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    modle_list=(MobileNetV2 MobileNetV1)
    for model_item in ${modle_list[@]}
    do
        echo "------------> begin to run ${model_item}"
        echo "index is speed, begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 1000 ${model_item} | tee ${log_path}/dynamic_${model_item}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 sp 1000 ${model_item} | tee ${log_path}/dynamic_${model_item}_mem_1gpus 2>&1
        sleep 60
        echo "index is mem, 1gpus, begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 2 sp 800 ${model_item} | tee ${log_path}/dynamic_${model_item}_speed_1gpus 2>&1
        sleep 60
        echo "index is mem, 8gpus, begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 2 sp 1000 ${model_item} | tee ${log_path}/dynamic_${model_item}_mem_1gpus 2>&1
    done
}
