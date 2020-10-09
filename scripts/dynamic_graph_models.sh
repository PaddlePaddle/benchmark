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

cur_model_list=(dy_yolov3 dy_mask_rcnn dy_slowfast dy_tsn dy_tsm dy_gan dy_seg dy_seq2seq dy_resnet dy_ptb_lm dy_transformer dy_mobilenet)

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
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp 1010 ${model_name} | tee ${log_path}/dynamic_${model_name}_speed_1gpus 2>&1
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
    model_list=(ResNet50_bs32 ResNet50_bs128)
    run_batchsize=32
    for model_item in ${model_list[@]}
    do
        if [ ${model_item} = "ResNet50_bs128" ]; then
            run_batchsize=128
        fi
        echo "index is speed, 1gpu begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 ${run_batchsize} ${model_item} sp 800 | tee ${log_path}/dynamic_${model_item}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${run_batchsize} ${model_item} mp 500 | tee ${log_path}/dynamic_${model_item}_speed_8gpus 2>&1
    done
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
    sleep 60
    echo "index is speed, 8gpus begin, mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 3000 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}

# tsn 
dy_tsn(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/tsn
    cd ${cur_model_path}

    pip install wget
    # Prepare pretrained modles
    ln -s ${prepare_path}/tsn/ResNet50_pretrained/ ${cur_model_path}/
    # Prepare data
    rm -rf data
    ln -s ${data_path}/dygraph_data/TSN/data ${cur_model_path}/

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 32 TSN sp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus begin, mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 32 TSN mp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}

# cyclegan and pix2pix
dy_gan(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleGAN
    cd ${cur_model_path}

    pip install tqdm
    # Prepare data
    mkdir -p data
    ln -s ${data_path}/dygraph_data/cityscapes_gan_mini ${cur_model_path}/data/cityscapes

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/gan_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(pix2pix cyclegan)
    for model_item in ${model_list[@]}
    do
        echo "index is speed, ${model_item} 1gpu begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp ${model_item} 1 | tee ${log_path}/dynamic_${model_item}_speed_1gpus 2>&1
        sleep 10
    done
}

#deeplabv3 and HRnet
dy_seg(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/dygraph
    cd ${cur_model_path}
    
    apt-get install lsb-core -y
    pip install  visualdl
    # Prepare data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/dygraph_data/cityscapes_hrnet_torch ${cur_model_path}/data/cityscapes
    
    # Running
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_list=(deeplabv3 HRnet)
    for model_item in ${model_list[@]}
    do
        echo "index is speed, ${model_item} 1gpu begin"
        CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh 1 sp ${model_item} 200 | tee ${log_path}/dynamic_${model_item}_speed_1gpus 2>&1
        sleep 10
        echo "index is speed, ${model_item} 8gpu begin"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp ${model_item} 200 | tee ${log_path}/dynamic_${model_item}_speed_8gpus 2>&1
        sleep 10
    done
}

dy_slowfast(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/slowfast
    cd ${cur_model_path}

    # Prepare data
    rm -rf data
    ln -s ${data_path}/dygraph_data/slowfast/data/ ${cur_model_path}/

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/slowfast/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 8 sp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus begin, mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 8 mp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}

dy_mask_rcnn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    git checkout dygraph
    pip install -r requirements.txt 

    # Install cocoapi
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
    if python -c "import tb_paddle" >/dev/null 2>&1;
    then
        echo "tb_paddle have already installed"
    else
        echo "tb_paddle NOT FOUND"
        pip install tb_paddle
        echo "tb_paddle installed"
    fi
    # Copy pretrained model
    ln -s ${prepare_path}/mask-rcnn/ResNet50_cos_pretrained  ~/.cache/paddle/weights
    cd ${cur_model_path}
    # Prepare data
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    # preprare scripts
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mask_rcnn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 sp 600 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus begin, mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}

dy_yolov3(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    git checkout dygraph
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
    echo "-------after data prepare"
    ls -l ./dataset/coco/
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpu, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
}

# tsm 
dy_tsm(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/tsm
    cd ${cur_model_path}

    pip install wget
    # Prepare pretrained modles
    ln -s ${prepare_path}/tsn/ResNet50_pretrained/ ${cur_model_path}/
    # Prepare data
    ln -s ${data_path}/dygraph_data/TSM/k400_wei/ ${cur_model_path}/
    ln -s ${data_path}/dygraph_data/TSM/ucf101 ${cur_model_path}/data/dataset/

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsm/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=5 bash run_benchmark.sh  1 16 TSM sp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus begin, mp"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 16 TSM mp 1 | tee ${log_path}/dynamic_${FUNCNAME}_speed_8gpus 2>&1
}
