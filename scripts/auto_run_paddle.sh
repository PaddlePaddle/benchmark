#!/bin/bash

cur_model_list=(seq2seq nextvlad detection mask_rcnn image_classification deeplab paddingrnn transformer CycleGAN  StarGAN STGAN Pix2pix bert yolov3)
usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  ${cur_model_list[@]} | all
  -c  cuda_version 9.0|10.0
  -n  image_name
  -i  image_commit_id
  -a  image_branch develop|1.6|pr_number|v1.6.0
  -v  paddle_version
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, such as /ssd1/ljh
  -t  job_type  benchmark_daliy | models test | pr_test
  -g  device_type  p40 | v100
  -s  implement_type of model static | dynamic
EOF
}
if [ $# -lt 18 ] ; then
  usage
  exit 1;
fi
while getopts h:m:c:n:i:a:v:p:t:g:s: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  n) image_name="$OPTARG" ;;
  i) image_commit_id="$OPTARG" ;;
  a) image_branch="$OPTARG" ;;
  v) paddle_version="$OPTARG" ;;
  p) all_path="$OPTARG" ;;
  t) job_type="$OPTARG" ;;
  g) device_type="$OPTARG" ;;
  s) implement_type="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

export https_proxy=http://172.19.57.45:3128
export http_proxy=http://172.19.57.45:3128

origin_path=$(pwd)

prepare(){
    echo "*******prepare benchmark***********"

#    this is for image paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
#    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
#    export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
#    yum install mysql-devel -y
#    pip install MySQL-python


#    this is for image paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
    if [ '10.0' = ${cuda_version} -o "p40" = ${device_type} ] ; then
        export LD_LIBRARY_PATH=/home/work/418.39/lib64/:$LD_LIBRARY_PATH
    fi

    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so
    rm /etc/apt/sources.list
    cp ${all_path}/sources.list /etc/apt
    apt-get update
    apt-get install libmysqlclient-dev -y
    apt-get install git -y
    pip install MySQL-python


    save_log_dir=${all_path}/logs/log_${paddle_version}

    if [[ -d ${save_log_dir} ]]; then
        rm -rf ${save_log_dir}
    fi

    train_log_dir=${save_log_dir}/train_log
    mkdir -p ${train_log_dir}

    export ROOT_PATH=/home/crim
    export BENCHMARK_ROOT=${ROOT_PATH}/benchmark
    log_path=${BENCHMARK_ROOT}/logs
    data_path=${all_path}/dataset
    prepare_path=${all_path}/prepare

    if [[ -e ${ROOT_PATH} ]]
    then
        rm ${log_path}/*
        cd ${ROOT_PATH}
        git pull
        echo "prepare had done"
    else
        mkdir /home/crim
        cd ${ROOT_PATH}
        git clone https://github.com/PaddlePaddle/benchmark.git
        cd ${BENCHMARK_ROOT}
        git submodule init
        git submodule update
        mkdir ${log_path}
    fi

    cd ${BENCHMARK_ROOT}
    benchmark_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "benchmark_commit_id is: "${benchmark_commit_id}
    pip uninstall paddlepaddle-gpu -y
    pip install ${image_name}
    echo "*******prepare end!***********"
}


#run_cycle_gan
CycleGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/cycle_gan
    cd ${cur_model_path}

    # Prepare data
    rm -rf  data/horse2zebra
    ln -s ${data_path}/horse2zebra/ ${cur_model_path}/data
    # Running ...
    rm ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/CycleGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run StartGAN
StarGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/
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
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run AttGAN
AttGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/
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
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run STGAN
STGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/
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
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run CGAN
CGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/
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
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed CGAN sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is mem, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem CGAN sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
        sleep 60
    done
}


#run Pix2pix
Pix2pix(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleGAN/
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
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed Pix2pix sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem Pix2pix sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run nextvlad
nextvlad(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleVideo/
    cd ${cur_model_path}

    # Prepare data
    rm -rf data
    mkdir -p data/dataset
    ln -s ${data_path}/youtube8m_paddle ./data/dataset/youtube8m

    # make train.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/train/* > ./data/dataset/youtube8m/train.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/val.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/test.list
    ls ${cur_model_path}/data/dataset/youtube8m/pkl/val/* > ./data/dataset/youtube8m/infer.list

    # Prepare package_list
    package_check_list=(imageio tqdm Cython pycocotools pandas wget)
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

    model_list=(nextvlad ctcn)
    for model_name in ${model_list[@]}; do
        echo "index is speed, 1gpu, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed 32 ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is mem, 1gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem 32 ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed 32 ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
        echo "index is mem, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem 32 ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_8gpus 2>&1
        sleep 60
    done
}


#run_deeplabv3+
deeplab(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/deeplabv3+
    cd ${cur_model_path}
    # Prepare data and pretrained parameters.
    mkdir data
    mkdir -p ./output/model
    ln -s ${data_path}/cityscape ${cur_model_path}/data/cityscape
    ln -s ${prepare_path}/deeplabv3plus_xception65_initialize ${cur_model_path}/deeplabv3plus_xception65_initialize
    # Running ...
    cp ${BENCHMARK_ROOT}/static_graph/deeplabv3+/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_mem_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_maxbs_1gpus 2>&1
    sleep 60
    echo "index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/DeepLab_V3+_maxbs_8gpus 2>&1

}


#run image_classification
image_classification(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/image_classification
    model_list=(SE_ResNeXt50_32x4d ResNet101 ResNet50)
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
    # running models cases
    for model_name in ${model_list[@]}; do
        echo "index is speed, 1gpu, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed 32 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed 32 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
        echo "index is mem, 1gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem 32 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
        sleep 60
        echo "index is mem, 8gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem 32 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_mem_8gpus 2>&1
        sleep 60
        echo "index is maxbs, 1gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs 112 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_maxbs_1gpus 2>&1
        sleep 60
        echo "index is maxbs, 8gpus, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs 112 ${model_name} sp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_maxbs_8gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed 32 ${model_name} mp 0 ${train_log_dir} ${profiler_log_dir} | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
        sleep 60
        echo "index is speed, 1gpu, is_profiler = 1, begin, ${model_name}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed 32 ${model_name} sp 1 ${train_log_dir} ${profiler_log_dir} | tee ${profiler_log_dir}/${model_name}_speed_1gpus 2>&1
        sleep 60
    done
}


#run_detection
detection(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/PaddleDetection
    cd ${cur_model_path}
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
    package_check_list=(imageio tqdm Cython pycocotools)
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
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is speed, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_8gpus 2>&1
        sleep 60
        echo "index is mem, 1gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
        sleep 60
        echo "index is mem, 8gpus, begin, ${model_name}"
        PYTHONPATH=$(pwd):${PYTHONPATH} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem ${model_name} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_8gpus 2>&1
        sleep 60
    done
}


#run_mask-rcnn
mask_rcnn(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/rcnn
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
    # Copy pretrained model
    ln -s ${prepare_path}/mask-rcnn/imagenet_resnet50_fusebn ${cur_model_path}/imagenet_resnet50_fusebn
    cd ${cur_model_path}
    # Prepare data
    ln -s ${data_path}/COCO17/annotations ${cur_model_path}/dataset/coco/annotations
    ln -s ${data_path}/COCO17/train2017 ${cur_model_path}/dataset/coco/train2017
    ln -s ${data_path}/COCO17/test2017 ${cur_model_path}/dataset/coco/test2017
    ln -s ${data_path}/COCO17/val2017 ${cur_model_path}/dataset/coco/val2017
    # Copy run_benchmark.sh and running ...
    cp ${BENCHMARK_ROOT}/static_graph/Mask-RCNN/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"

    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_maxbs_1gpus 2>&1
    sleep 60
    echo "index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_maxbs_8gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed mp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_8gpus8p 2>&1
}


#run_bert
bert(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/language_representations_kit/BERT
    cd ${cur_model_path}
    rm -rf data
    ln -s ${data_path}/Bert/data ${cur_model_path}/data
    ln -s ${data_path}/Bert/MNLI ${cur_model_path}/MNLI
    ln -s ${prepare_path}/Bert/chinese_L-12_H-768_A-12 ${cur_model_path}/chinese_L-12_H-768_A-12
    ln -s ${prepare_path}/Bert/uncased_L-24_H-1024_A-16 ${cur_model_path}/uncased_L-24_H-1024_A-16
    cp ${BENCHMARK_ROOT}/static_graph/NeuralMachineTranslation/BERT/fluid/run_benchmark.sh ./run_benchmark.sh

    sed -i '/set\ -xe/d' run_benchmark.sh

    model_mode_list=(base large)
    fp_mode_list=(fp32 fp16)
    for model_mode in ${model_mode_list[@]}; do
        for fp_mode in ${fp_mode_list[@]}; do
            model_name="${FUNCNAME}_${model_mode}_${fp_mode}"
            echo "index is speed, 1gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
            sleep 60
            echo "index is speed, 8gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_8gpus 2>&1
            sleep 60
            echo "index is mem, 1gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
            sleep 60
            echo "index is mem, 8gpus, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_8gpus 2>&1
            sleep 60
            #echo "index is maxbs, 1gpus, begin, ${model_name}"
            #CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_maxbs_1gpus 2>&1
            #sleep 60
            #echo "index is maxbs, 8gpus, begin, ${model_name}"
            #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs ${model_mode} ${fp_mode} sp ${train_log_dir} | tee ${log_path}/${model_name}_maxbs_8gpus 2>&1
            #sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_mode} ${fp_mode} mp ${train_log_dir} | tee ${log_path}/${model_name}_speed_8gpus8p 2>&1
            sleep 60
        done
    done
}


#run_transformer
transformer(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/neural_machine_translation/transformer
    cd ${cur_model_path}
    ln -s ${data_path}/transformer/data ${cur_model_path}/data
    cp -r ${prepare_path}/transformer/mosesdecoder ${cur_model_path}/mosesdecoder
    cp ${BENCHMARK_ROOT}/static_graph/NeuralMachineTranslation/Transformer/fluid/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_type="base"
    echo "model_type is ${model_type}, index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_mem_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_mem_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_type} mp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus8p 2>&1
    sleep 60
    model_type="big"
    echo "model_type is ${model_type}, index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_mem_1gpus 2>&1
    sleep 60
    echo "model_type is ${model_type}, index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_mem_8gpus 2>&1
    sleep 60
#    echo "model_type is ${model_type}, index is maxbs, 1gpus, begin"
#    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_1gpus 2>&1
#    sleep 60
#    echo "model_type is ${model_type}, index is maxbs, 8gpus, begin"
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs ${model_type} sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_maxbs_8gpus 2>&1
#    sleep 60
    echo "model_type is ${model_type}, index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed ${model_type} mp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_${model_type}_speed_8gpus8p 2>&1
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
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
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
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed ${model_type} ${rnn_type} sp ${train_log_dir} | tee ${log_path}/${model_name}_speed_1gpus 2>&1
        sleep 60
        echo "index is mem, 1gpus, ${model_name}, begin"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem ${model_type} ${rnn_type} sp ${train_log_dir} | tee ${log_path}/${model_name}_mem_1gpus 2>&1
        sleep 60
        done
    done
}


#run_yolov3
yolov3(){
    cur_model_path=${BENCHMARK_ROOT}/yolov3/paddle
    cd ${cur_model_path}
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

#    cd ${cur_model_path}
#    #yolov3 的模型代码还在models
#    git clone https://github.com/PaddlePaddle/models.git
#    cd models/PaddleCV/yolov3/

    cd ${BENCHMARK_ROOT}/models/PaddleCV/yolov3/
    #git checkout -b benchmark origin/benchmark

    sed -i 's/build_strategy.memory_optimize/#build_strategy.memory_optimize/g' train.py
    #sh ./weights/download.sh
    ln -s ${prepare_path}/yolov3/yolov3 ./weights/yolov3
    ln -s ${prepare_path}/yolov3/darknet53 ./weights/darknet53

    rm -rf dataset/coco
    ln -s ${data_path}/coco ./dataset/coco
    cp ${BENCHMARK_ROOT}/static_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_8gpus 2>&1
    sleep 60
    echo "index is maxbs, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_maxbs_1gpus 2>&1
    sleep 60
    echo "index is maxbs, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh maxbs sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_maxbs_8gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, run_mode is multi_process, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh speed mp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_8gpus8p 2>&1
}


#run seq2seq
seq2seq(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleNLP/unarchived/neural_machine_translation/rnn_search
    cd ${cur_model_path}

    # Prepare data
    rm -r ${cur_model_path}/data
    mkdir ${cur_model_path}/data
    ln -s ${data_path}/seq2seq_paddle/en-vi ${cur_model_path}/data

    # Running ...
    cp ${BENCHMARK_ROOT}/static_graph/seq2seq/paddle/run_benchmark.sh ./

    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh speed sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh mem sp ${train_log_dir} | tee ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


run(){
    if [ $model = "all" ]
    then
        for model_name in ${cur_model_list[@]}
        do
            echo "=====================${model_name} run begin=================="
            $model_name
            sleep 60
            echo "*********************${model_name} run end!!******************"
        done
    elif echo ${cur_model_list[@]} | grep -w $model &>/dev/null
    then
        echo "=====================${model} run begin=================="
        $model
        sleep 60
        echo "*********************${model} run end!!******************"
    else
        echo "model: $model not support!"
    fi
}


save(){
    mv ${log_path} ${save_log_dir}/index
    ln -s ${all_path}/env/bin/python /usr/local/bin/mypython
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${all_path}/env/lib/
    cd ${origin_path}

    echo "==================== begin insert to sql ================="
    echo "benchmark_commit_id = ${benchmark_commit_id}"
    echo "   paddle_commit_id = ${image_commit_id}"
    echo "      paddle_branch = ${image_branch}"
    echo "     implement_type = ${implement_type}"
    echo "     paddle_version = ${paddle_version}"
    echo "       cuda_version = ${cuda_version}"
    echo "           log_path = ${save_log_dir}"
    echo "           job_type = ${job_type}"
    echo "           device_type = ${device_type}"

    mypython save.py --code_commit_id ${benchmark_commit_id} \
                 --image_commit_id ${image_commit_id} \
                 --image_branch ${image_branch} \
                 --log_path ${save_log_dir} \
                 --cuda_version ${cuda_version} \
                 --paddle_version ${paddle_version} \
                 --job_type ${job_type} \
                 --device_type ${device_type} \
                 --implement_type ${implement_type}

    echo "******************** end insert to sql!! *****************"
}

prepare
run
save

