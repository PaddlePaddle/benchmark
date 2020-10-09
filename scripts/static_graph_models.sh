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

cur_model_list=(detection mask_rcnn seq2seq nextvlad image_classification deeplab paddingrnn transformer CycleGAN  StarGAN STGAN Pix2pix bert yolov3)

#run_cycle_gan
CycleGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/horse2zebra/ ${cur_model_path}/data/cityscapes
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/CycleGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, profiler is on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
}


#run StartGAN
StarGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/

    # Install imageio
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio have already installed"
    else
        echo "imageio NOT FOUND"
        pip install imageio
        echo "imageio installed"
    fi
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/StarGAN/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, profiler is on,  begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
}


#run AttGAN
AttGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/

    # Install imageio
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio have already installed"
    else
        echo "imageio NOT FOUND"
        pip install imageio
        echo "imageio installed"
    fi
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/AttGAN/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
}


#run STGAN
STGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data/celeba
    ln -s ${data_path}/CelebA/Anno/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Eval/* ${cur_model_path}/data/celeba/
    ln -s ${data_path}/CelebA/Img/img_align_celeba ${cur_model_path}/data/celeba/

    # Install imageio
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio have already installed"
    else
        echo "imageio NOT FOUND"
        pip install imageio
        echo "imageio installed"
    fi
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/STGAN/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, profiler is on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
}


#run CGAN
CGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/mnist ${cur_model_path}/data

    # Install imageio
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio have already installed"
    else
        echo "imageio NOT FOUND"
        pip install imageio
        echo "imageio installed"
    fi
    # cp run_benchmark.sh
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/GAN_models/PaddleGAN/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh

    # running models cases
    model_list=(CGAN DCGAN)
    for model_name in ${model_list[@]}; do
        echo "index is speed, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 CGAN sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
    done
}


#run Pix2pix
Pix2pix(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir ${cur_model_path}/data
    ln -s ${data_path}/Pix2pix/cityscapes ${cur_model_path}/data

    # Install imageio
    if python -c "import imageio" >/dev/null 2>&1
    then
        echo "imageio have already installed"
    else
        echo "imageio NOT FOUND"
        pip install imageio
        echo "imageio installed"
    fi
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/GAN_models/PaddleGAN/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh

    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 Pix2pix sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, profiler is on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 Pix2pix sp 300 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
}


#run nextvlad
nextvlad(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/video/
    cd ${cur_model_path}

    # Prepare data
    rm -rf data
    mkdir -p data/dataset
    ln -s ${data_path}/youtube8m_paddle ./data/dataset/youtube8m
    ln -s ${data_path}/ctcn_paddle/ ./data/dataset/ctcn

    # make train.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/train/* > ./data/dataset/youtube8m/train.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/val.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/test.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/infer.list

    # Prepare package_list
    package_check_list=(imageio tqdm Cython pycocotools pandas wget h5py)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
        else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
        fi
    done

    #Running
    cp ${BENCHMARK_ROOT}/static_graph/NextVlad/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh

    model_list=(nextvlad) # CTCN)
    for model_name in ${model_list[@]}; do
        echo "index is speed, 1gpu, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 32 ${model_name} sp 2 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 1gpu, prfoiler is on, begin, ${model_name}"
#        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 32 ${model_name} sp 2 | tee ${log_path}/${model_name}_speed_1gpus_profiler 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 32 ${model_name} sp 2 | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
    done
}


#run_deeplabv3+
deeplab(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg
    cd ${cur_model_path}
    # Prepare data and pretrained parameters.
    ln -s ${data_path}/cityscape ${cur_model_path}/dataset/cityscapes
    ln -s ${prepare_path}/deeplabv3p_xception65_bn_cityscapes ${cur_model_path}/pretrained_model/
    # Running ...
    cp ${BENCHMARK_ROOT}/static_graph/deeplabv3+/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 2 | tee ${log_path}/DeepLab_V3+_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 1gpu, profiler is on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 1 | tee ${log_path}/DeepLab_V3+_speed_1gpus_profiler 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 sp 1 | tee ${log_path}/DeepLab_V3+_speed_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 sp 1 | tee ${log_path}/DeepLab_V3+_maxbs_1gpus 2>&1
    sleep 60
    echo "index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 sp 1 | tee ${log_path}/DeepLab_V3+_maxbs_8gpus 2>&1

}


#run image_classification
image_classification(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/image_classification
    cd ${cur_model_path}
    # Prepare data
    ln -s ${data_path}/ILSVRC2012/train ${cur_model_path}/data/ILSVRC2012/train
    ln -s ${data_path}/ILSVRC2012/train_list.txt ${cur_model_path}/data/ILSVRC2012/train_list.txt
    ln -s ${data_path}/ILSVRC2012/val ${cur_model_path}/data/ILSVRC2012/val
    ln -s ${data_path}/ILSVRC2012/val_list.txt ${cur_model_path}/data/ILSVRC2012/val_list.txt
    # Copy run_benchmark.sh and running ...
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/cd /d' run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
    # running models cases
    model_list=(SE_ResNeXt50_32x4d ResNet101 ResNet50_bs32 ResNet50_bs128)
    run_batchsize=32
    for model_name in ${model_list[@]}; do
        if [ ${model_name} = "ResNet50_bs128" ]; then
            run_batchsize=128
        fi
        echo "index is speed, 1gpu, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${run_batchsize} ${model_name} sp 800 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 1gpu, begin, profile is on, ${model_name}"
#        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${run_batchsize} ${model_name} sp 800 | tee ${log_path}/${model_name}_speed_1gpus_profiler 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${run_batchsize} ${model_name} sp 500 | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
        echo "index is maxbs, 1gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 112 ${model_name} sp 500 | tee ${log_path}/${model_name}_maxbs_1gpus 2>&1
        sleep 60
        #echo "index is maxbs, 8gpus, begin, ${model_name}"
        #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 112 ${model_name} sp 500 | tee ${log_path}/${model_name}_maxbs_8gpus 2>&1
        #sleep 60
        echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${run_batchsize} ${model_name} mp 1000 | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
        sleep 60
    done
}


#run_detection
detection(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    ## test dir
    git branch
    ## ls 
    ls
    ## ls tools
    ls tools
    ## mkdir dataset 
    rm -rf dataset/coco/
    mkdir -p dataset/coco/
    # Prepare data
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    #prepare pretrain_models
    ln -s ${prepare_path}/detection/ResNet101_vd_pretrained ~/.cache/paddle/weights
    ln -s ${prepare_path}/detection/ResNet50_cos_pretrained ~/.cache/paddle/weights
    ln -s ${prepare_path}/detection/ResNeXt101_vd_64x4d_pretrained ~/.cache/paddle/weights

    # Prepare package_list
    package_check_list=(imageio tqdm Cython pycocotools tb_paddle)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
        else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
        fi
    done

    # Copy run_benchmark.sh and running ...
    cp ${BENCHMARK_ROOT}/static_graph/Detection/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh

    model_list=(mask_rcnn_fpn_resnet mask_rcnn_fpn_resnext retinanet_rcnn_fpn cascade_rcnn_fpn)
    for model_name in ${model_list[@]}; do
        echo "index is speed, 1gpu, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_name} sp 600 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 1gpu, profiler is on  begin, ${model_name}"
#        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${model_name} sp 600 | tee ${log_path}/${model_name}_speed_1gpus_profiler 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_name} sp 600 | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
    done
}


#run_mask-rcnn
mask_rcnn(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}

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
        python2 setup.py install --user
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
    # Copy run_benchmark.sh and running ...
    cp ${BENCHMARK_ROOT}/static_graph/Mask-RCNN/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"

    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 1gpu, profiler is on begin"

#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 sp 200 | tee ${log_path}/${FUNCNAME}_maxbs_1gpus 2>&1
    sleep 60
    echo "index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 sp 200 | tee ${log_path}/${FUNCNAME}_maxbs_8gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/${FUNCNAME}_speed_8gpus8p 2>&1
}


#run_bert
bert(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/pretrain_language_models/BERT/
    cd ${cur_model_path}
    rm -rf data
    ln -s ${data_path}/Bert/data ${cur_model_path}/data
    ln -s ${data_path}/Bert/MNLI ${cur_model_path}/MNLI
    ln -s ${prepare_path}/Bert/chinese_L-12_H-768_A-12 ${cur_model_path}/chinese_L-12_H-768_A-12
    ln -s ${prepare_path}/Bert/uncased_L-24_H-1024_A-16 ${cur_model_path}/uncased_L-24_H-1024_A-16
    cp ${BENCHMARK_ROOT}/static_graph/BERT/paddle/run_benchmark.sh ./run_benchmark.sh

    sed -i '/set\ -xe/d' run_benchmark.sh

    model_mode_list=(base large)
    fp_mode_list=(fp32 fp16)
    for model_mode in ${model_mode_list[@]}; do
        for fp_mode in ${fp_mode_list[@]}; do
            model_name="${FUNCNAME}_${model_mode}_${fp_mode}"
            echo "index is speed, 1gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} sp 1500 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
            sleep 60
            echo "index is speed, 1gpus, profiler is on, begin, ${model_name}"
#            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${model_mode} ${fp_mode} sp 1500 | tee ${log_path}/${model_name}_speed_1gpus_profiler 2>&1
            sleep 60
            echo "index is speed, 8gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} sp 500 | tee ${log_path}/${model_name}_speed_8gpus 2>&1
            sleep 60
            #echo "index is maxbs, 1gpus, begin, ${model_name}"
            #CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_maxbs_1gpus 2>&1
            #sleep 60
            #echo "index is maxbs, 8gpus, begin, ${model_name}"
            #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_maxbs_8gpus 2>&1
            #sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp 800 | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
            sleep 60
        done
    done
}


#run_transformer
transformer(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/machine_translation/transformer/
    cd ${cur_model_path}
    ln -s ${data_path}/transformer/data ${cur_model_path}/data
    cp -r ${prepare_path}/transformer/mosesdecoder ${cur_model_path}/mosesdecoder
    cp ${BENCHMARK_ROOT}/static_graph/Transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_type="base"
    echo "model_type is ${model_type}, index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed,profiler is on, 1gpu, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus_profiler 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 ${model_type} sp 400 | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 ${model_type} sp 400 | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_type} mp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus8p 2>&1
    sleep 60
    model_type="big"
    echo "model_type is ${model_type}, index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, profiler is on, 1gpu, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus_profiler 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_type} sp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 ${model_type} sp 400 | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 ${model_type} sp 400 | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 ${model_type} mp 600 | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus8p 2>&1
}


#run_ddpg_deep_explore
ddpg_deep_explore(){
    cur_model_path=${BENCHMARK_ROOT}/DDPG_Deep_Explore/Fluid_version
    cd ${cur_model_path}
    if python -c "import parl" >/dev/null 2>&1
    then
        echo "parl have already installed"
    else
        echo "parl NOT FOUND"
        pip install parl==1.1
        echo "parl installed"
    fi
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
}


#run_paddingrnn
paddingrnn(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/language_model
    cd ${cur_model_path}
    # Prepare data.
    ln -s ${data_path}/simple-examples ${cur_model_path}/data/simple-examples
    # Running ...
    cp ${BENCHMARK_ROOT}/static_graph/PaddingRNN/lstm_paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_type_list=(small large)
    rnn_type_list=(static padding)
    for model_type in ${model_type_list[@]}; do
        for rnn_type in ${rnn_type_list[@]}; do
        model_name="${FUNCNAME}_${model_type}_${rnn_type}"
        echo "index is speed, 1gpus, ${model_name}, begin"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 ${model_type} ${rnn_type} sp 3 | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 1gpus, ${model_name}, profiler is on, begin"
#        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 ${model_type} ${rnn_type} sp 1 | tee ${log_path}/${model_name}_speed_1gpus_profiler 2>&1
        sleep 60
        done
    done
}


#run_yolov3
yolov3(){
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "cocoapi have already installed"
    else
        echo "cocoapi NOT FOUND"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython
        make install
        python2 setup.py install --user
        echo "cocoapi installed"
    fi

    cd ${BENCHMARK_ROOT}/PaddleDetection

    #sh ./weights/download.sh
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    rm -rf dataset/coco
    ln -s ${data_path}/coco ./dataset/coco
    cp ${BENCHMARK_ROOT}/static_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 1gpu, profiler on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 600 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 sp 600 | tee ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 6 sp 600 | tee ${log_path}/${FUNCNAME}_maxbs_1gpus 2>&1
    sleep 60
    #echo "index is maxbs, 8gpus, begin"
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 6 sp 600 | tee ${log_path}/${FUNCNAME}_maxbs_8gpus 2>&1
    #sleep 60
    echo "index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh 1 mp 600 | tee ${log_path}/${FUNCNAME}_speed_8gpus8p 2>&1
}


# seq2seq
seq2seq(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/seq2seq/seq2seq/
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir ${cur_model_path}/data
    ln -s ${data_path}/seq2seq_paddle/en-vi ${cur_model_path}/data

    # Running ...
    cp ${BENCHMARK_ROOT}/static_graph/seq2seq/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 1 | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, profiler is on, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 3 sp 1 | tee ${log_path}/${FUNCNAME}_speed_1gpus_profiler 2>&1
}
