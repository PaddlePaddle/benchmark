#!/bin/bash

cur_model_list=(CycleGAN deeplab se_resnext50 mask_rcnn bert transformer ddpg_deep_explore paddingrnn yolov3)
usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  ${cur_model_list[@]} | all
EOF
}
if [ $# != 2 ] ; then
  usage
  exit 1;
fi
while getopts h:m: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128

prepare(){
    echo "*******prepare benchmark***********"

    root_path=/home/crim/
    fluid_path=/home/crim/benchmark
    log_path=/home/crim/benchmark/logs
    data_path=/ssd1/ljh/dataset
    prepare_path=/ssd1/ljh/prepare

    if [ -e ${root_path} ]
    then
        rm ${log_path}/*
        cd ${root_path}
        git pull
        echo "prepare had done"
    else
        mkdir /home/crim
        cd ${root_path}
        git clone https://github.com/PaddlePaddle/benchmark.git
        cd ${fluid_path}
        git submodule init
        git submodule update
        mkdir $log_path
    fi

    echo "*******prepare end!***********"
}

#run_cycle_gan
CycleGAN(){
    cur_model_path=${fluid_path}/CycleGAN/paddle
    cd ${cur_model_path}
    mkdir data
    ln -s ${data_path}/horse2zebra/ ${cur_model_path}/data
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run_deeplabv3+
deeplab(){
    cur_model_path=${fluid_path}/deeplabv3+/paddle
    cd ${cur_model_path}
    mkdir data
    mkdir -p ./output/model
    ln -s ${data_path}/cityscape ${cur_model_path}/data/cityscape
    ln -s ${prepare_path}/deeplabv3plus_xception65_initialize ${cur_model_path}/deeplabv3plus_xception65_initialize
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh speed > ${log_path}/DeepLab_V3+_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 4gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh speed > ${log_path}/DeepLab_V3+_speed_4gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh mem > ${log_path}/DeepLab_V3+_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 4gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh mem > ${log_path}/DeepLab_V3+_mem_4gpus 2>&1
}


#run_se-resnext50
se_resnext50(){
    cur_model_path=${fluid_path}/se-resnext/paddle
    cd ${cur_model_path}
    rm -rf ${cur_model_path}/data/ILSVRC2012
    ln -s ${data_path}/ILSVRC2012 ${cur_model_path}/data/ILSVRC2012
    sed -i '/cd /d' run.sh
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh speed 32 > ${log_path}/SE-ResNeXt50_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh speed 32 > ${log_path}/SE-ResNeXt50_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh mem 32 > ${log_path}/SE-ResNeXt50_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh mem 32 > ${log_path}/SE-ResNeXt50_mem_8gpus 2>&1
}


#run_mask-rcnn
mask_rcnn(){
    cur_model_path=${fluid_path}/Mask-RCNN/paddle
    cd ${cur_model_path}
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "cocoapi have already installed"
    else
        echo "cocoapi NOT FOUND"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython > ${log_path}/${FUNCNAME}_speed 2>&1
        make install > ${log_path}/${FUNCNAME}_speed 2>&1
        python2 setup.py install --user > ${log_path}/${FUNCNAME}_speed 2>&1
        echo "cocoapi installed"
    fi
    ln -s ${prepare_path}/mask-rcnn/imagenet_resnet50_fusebn ${cur_model_path}/imagenet_resnet50_fusebn
    cd rcnn
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ${cur_model_path}/rcnn/dataset/coco
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_8gpus 2>&1
}


#run_bert
bert(){
    cur_model_path=${fluid_path}/NeuralMachineTranslation/BERT/fluid/train
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/data ${cur_model_path}/data
    ln -s ${prepare_path}/chinese_L-12_H-768_A-12 ${cur_model_path}/chinese_L-12_H-768_A-12

    echo "index is speed, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_8gpus 2>&1
}


#run_transformer
transformer(){
    cur_model_path=${fluid_path}/NeuralMachineTranslation/Transformer/fluid/train
    cd ${cur_model_path}
    ln -s ${data_path}/transformer/data ${cur_model_path}/data
    cp -r ${prepare_path}/transformer/mosesdecoder ${cur_model_path}/mosesdecoder
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_8gpus 2>&1
}


#run_ddpg_deep_explore
ddpg_deep_explore(){
    cur_model_path=${fluid_path}/DDPG_Deep_Explore/Fluid_version
    cd ${cur_model_path}
    if python -c "import parl" >/dev/null 2>&1
    then
        echo "parl have already installed"
    else
        echo "parl NOT FOUND"
        pip install parl > ${log_path}/${FUNCNAME}_speed 2>&1
        echo "parl installed"
    fi
    echo "index is speed, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
}


#run_paddingrnn
paddingrnn(){
    cur_model_path=${fluid_path}/PaddingRNN/lstm_paddle
    cd ${cur_model_path}
    batch_size=20
    ln -s ${data_path}/simple-examples ${cur_model_path}/data/simple-examples
    echo "index is speed, 1gpus, small model, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh speed small ${batch_size} > ${log_path}/${FUNCNAME}_small_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, small model, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh mem small ${batch_size} > ${log_path}/${FUNCNAME}_small_mem_1gpus 2>&1
    echo "index is speed, 1gpus, large model, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh speed large ${batch_size} > ${log_path}/${FUNCNAME}_large_speed_1gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, large model, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh mem large ${batch_size} > ${log_path}/${FUNCNAME}_large_mem_1gpus 2>&1
}


#run_yolov3
yolov3(){
    cur_model_path=${fluid_path}/yolov3/paddle
    cd ${cur_model_path}
    if python -c "import pycocotools" >/dev/null 2>&1
    then
        echo "cocoapi have already installed"
    else
        echo "cocoapi NOT FOUND"
        cp -r ${prepare_path}/cocoapi/ ./
        cd cocoapi/PythonAPI/
        pip install Cython > ${log_path}/${FUNCNAME}_speed 2>&1
        make install > ${log_path}/${FUNCNAME}_speed 2>&1
        python2 setup.py install --user > ${log_path}/${FUNCNAME}_speed 2>&1
        echo "cocoapi installed"
    fi
    #yolov3 的模型代码还在models
    git clone https://github.com/PaddlePaddle/models.git
    cd models/PaddleCV/yolov3/

    #sh ./weights/download.sh
    ln -s ${prepare_path}/yolov3/yolov3 ./weights/yolov3
    ln -s ${prepare_path}/yolov3/darknet53 ./weights/darknet53

    rm -rf dataset/coco
    ln -s ${data_path}/coco ./dataset/coco
    cp ${fluid_path}/yolov3/paddle/run.sh ./
    echo "index is speed, 1gpu, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train speed > ${log_path}/${FUNCNAME}_speed_8gpus 2>&1
    sleep 60
    echo "index is mem, 1gpus, begin"
    CUDA_VISIBLE_DEVICES=0 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_1gpus 2>&1
    sleep 60
    echo "index is mem, 8gpus, begin"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train mem > ${log_path}/${FUNCNAME}_mem_8gpus 2>&1

}

prepare

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
