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

cur_model_list=(dy_to_static_yolov3 dy_to_static_resnet dy_to_static_mobilenet)

# MobileNet
dy_to_static_mobilenet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt

    # Prepare data
    ln -s ${data_path}/dygraph_data/imagenet100_data ${cur_model_path}/dataset/         # 准备数据集,需>要保证benchmark任务极其21 上对应目录下存在该数据集！

    # Running ...
    rm -f ./run_benchmark_mobilenet.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/mobilenet/paddle/run_benchmark_mobilenet.sh ./       # 拷贝脚本到当前目录
    sed -i '/set\ -xe/d' run_benchmark_mobilenet.sh
    modle_list=(MobileNetV2 MobileNetV1)
    for model_name in ${modle_list[@]}
    do
        echo "------------> begin to run ${model_name}"
        echo "index is speed, 1gpu begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark_mobilenet.sh 1  sp 1  ${model_name} | tee ${log_path}/dynamic_${model_name}_bs128_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_mobilenet.sh 1  mp  1 ${model_name} | tee ${log_path}/dynamic_${model_name}_bs128_speed_8gpus 2>&1
        sleep 60
    done
}

dy_to_static_yolov3(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch    #develop 分支
    cd ${cur_model_path}
    pip install Cython
    pip install -r requirements.txt 
   
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "cocoapi have already installed"
    else
        echo "cocoapi NOT FOUND"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython
        make install
        python setup.py install --user
        echo "cocoapi installed"
    fi

    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    cd ${cur_model_path}
    echo "-------before data prepare"
    ls -l ./dataset/coco/
    ln -s ${data_path}/coco/* ./dataset/coco/
    pip install -r ./dataset/coco/requirements.txt
    echo "-------after data prepare"
    ls -l ./dataset/coco/
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 500 | tee ${log_path}/dynamic_yolov3_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpu, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 500 | tee ${log_path}/dynamic_yolov3_speed_8gpus 2>&1
}

dy_to_static_resnet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    pip install -r requirements.txt
   
    ln -s ${data_path}/dygraph_data/imagenet100_data/ ${cur_model_path}/dataset
    rm -f ./run_benchmark_resnet.sh
    cp ${BENCHMARK_ROOT}/dynamic_to_static/resnet/paddle/run_benchmark_resnet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_resnet.sh
    batch_size=32
    model_list=(ResNet152_bs32 ResNet50_bs32 ResNet50_bs128)
    for model_name in ${model_list[@]}
    do
        if [ ${model_name} == "ResNet50_bs128" ]; then
            batch_size=128
        fi
        echo "model is ${model_name}, index is speed, 1gpu begin"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} sp 1 | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "model is ${model_name}, index is speed, 8gpu begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark_resnet.sh 1 ${batch_size} ${model_name} mp 1 | tee ${log_path}/dynamic_${model_name}_speed_8gpus 2>&1
        sleep 60
    done
}
