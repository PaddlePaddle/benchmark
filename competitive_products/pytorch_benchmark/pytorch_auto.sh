#!/bin/bash

##########################################
#                                        #
#      usage                             #
#      export the BENCHMARK_ROOT         #
#      export the PYTORCH_BENCHMARK_ROOT #
#                                        #
##########################################

#************ note that you neet the images of pytorch which name contains devel***********#

cur_model_list=(detection pix2pix stargan image_classification dy_image_class dy_sequence dy_ptb dy_gan)

######################
environment(){
    export LD_LIBRARY_PATH=/home/work/418.39/lib64/:/usr/local/cuda-10.0/compat/:$LD_LIBRARY_PATH
    apt-get update
    apt-get install wget vim git libglib2.0-dev apt-file psmisc net-tools -y
    apt-file update
    apt-get install libsm6 libxrender1 libxext-dev -y
    
    pip uninstall torch-nightly -y
    #conda remove wrapt --y
    #pip uninstall setuptools -y
    #pip install setuptools>=41.0.0
    pip uninstall tensorflow  tensorboard tensorflow-estimator Pillow -y
    pip install tensorflow==1.14.0
    pip install Pillow==6.1
    pip install transformers
    
    package_check_list=(pytest Cython opencv-python future pycocotools matplotlib networkx fasttext visdom protobuf dominate enum)
        for package in ${package_check_list[@]}; do
            if python -c "import ${package}" >/dev/null 2>&1; then
                echo "${package} have already installed"
            else
                echo "${package} NOT FOUND"
                pip install ${package}
                echo "${package} installed"
            fi
    done
    
    cd /usr/bin/
    ln -s /ssd3/heya/tools/monquery
    ln -s /ssd3/heya/tools/monqueryfunctions
    export LD_LIBRARY_PATH=/home/work/418.39/lib64/:${LD_LIBRARY_PATH}

}


#################pip packages
prepare(){
    export BENCHMARK_ROOT=/ssd3/heya/pytorch/0430_cuda10/benchmark/
    export PYTORCH_BENCHMARK_ROOT=${BENCHMARK_ROOT}/competitive_products/pytorch_benchmark
    
    export datapath=/ssd1/ljh/dataset
    
    cur_timestaps=$(date "+%Y%m%d%H%M%S")
    export CUR_DIR=${PYTORCH_BENCHMARK_ROOT}/${cur_timestaps}_result/
    export LOG_DIR=${CUR_DIR}/LOG
    export TRAIN_LOG_DIR=${CUR_DIR}/LOG_DY
    export RES_DIR=${CUR_DIR}/RES
    export RES_DIR_DY=${CUR_DIR}/RES_DY
    export MODEL_PATH=${CUR_DIR}/py_models
    
    mkdir -p ${LOG_DIR} ${TRAIN_LOG_DIR}
    mkdir -p ${RES_DIR} ${RES_DIR_DY}
    mkdir -p ${MODEL_PATH}

}

#########detection
detection(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    
    cp -r ${BENCHMARK_ROOT}/static_graph/Detection/pytorch ${curl_model_path}/pytorch_detection
    cd ${curl_model_path}/pytorch_detection
    ln -s ${datapath}/COCO17 ./Detectron/detectron/datasets/data/coco
    rm ${curl_model_path}/pytorch_detection/run_detectron.sh
    cp ${BENCHMARK_ROOT}/static_graph/Detection/pytorch/run_detectron.sh ${curl_model_path}/pytorch_detection/
    
    model_list=(mask_rcnn_fpn_resnet mask_rcnn_fpn_resnext retinanet_rcnn_fpn cascade_rcnn_fpn)
    # maybe you need py2 env to run cascade_rcnn_fpn because of the compatibility
    for model_name in ${model_list[@]}; do
        echo "----------------${model_name}"
        echo "------1-----------}"
        CUDA_VISIBLE_DEVICES=0 bash run_detectron.sh 1 ${model_name} sp ${LOG_DIR} > ${RES_DIR}/${model_name}_1.res 2>&1
        sleep 60
        echo "------8-----------"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_detectron.sh 1 ${model_name} sp ${LOG_DIR} > ${RES_DIR}/${model_name}_8.res 2>&1
        sleep 60
    done 
}

######################pix2pix
pix2pix(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    echo "--1"${curl_model_path}
    echo "--2"${PYTORCH_BENCHMARK_ROOT}
    
    git clone https://github.com/chengduoZH/pytorch-CycleGAN-and-pix2pix
    
    echo "git success"
    cp ${BENCHMARK_ROOT}/static_graph/GAN_models/PytorchGAN/run.sh ${curl_model_path}/pytorch-CycleGAN-and-pix2pix/run_pix2pix.sh
    cd ${curl_model_path}/pytorch-CycleGAN-and-pix2pix
    git checkout benchmark
    ln -s ${datapath}/pytorch_pix2pix_data ${curl_model_path}/pytorch-CycleGAN-and-pix2pix/dataset
    
    echo "----------------pix2pix"
        echo "----1----}"
    CUDA_VISIBLE_DEVICES=0 bash run_pix2pix.sh 1 ${LOG_DIR} > ${RES_DIR}/pix2pix_1.res 2>&1
    #sleep 60
    #    echo "----8----}" # not run
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_pix2pix.sh 1 ${LOG_DIR} > ${RES_DIR}/pix2pix_8.res 2>&1

}


##################stargan
stargan(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    
    git clone https://github.com/yunjey/stargan.git
    cd ${curl_model_path}/stargan
    cp ${BENCHMARK_ROOT}/static_graph/StarGAN/pytorch/run_stargan.sh ${curl_model_path}/stargan 
    mkdir -p ${curl_model_path}/stargan/data/celeba
    ln -s ${datapath}/CelebA/Anno/* ${curl_model_path}/stargan/data/celeba
    ln -s ${datapath}/CelebA/Img/img_align_celeba/ ${curl_model_path}/stargan/data/celeba/images
    echo "----------------stargan"
        echo "----1----}"
    CUDA_VISIBLE_DEVICES=0 bash run_stargan.sh train 1 ${LOG_DIR} > ${RES_DIR}/stargan_1.res 2>&1
    #sleep 60
    #    echo "----8----}" # not run
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_stargan.sh train 1 ${LOG_DIR} > ${RES_DIR}/stargan_8.res 2>&1
}


#######################image_class
image_classification(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    
    cp -r ${BENCHMARK_ROOT}/static_graph/image_classification/pytorch ${curl_model_path}/pytorch_image_class
    cd ${curl_model_path}/pytorch_image_class
    ln -s /ssd3/ljh/cts_ce/dataset/data/ImageNet/train ${curl_model_path}/pytorch_image_class/SENet/ImageData/ILSVRC2012_img_train
    ln -s /ssd3/ljh/cts_ce/dataset/data/ImageNet/val ${curl_model_path}/pytorch_image_class/SENet/ImageData/ILSVRC2012_img_val
    
    
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/pytorch/run_vision.sh ${curl_model_path}/pytorch_image_class
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/pytorch/run_senet.sh ${curl_model_path}/pytorch_image_class
    
    
    model_list=(resnet101 resnet50)
    for model_name in ${model_list[@]}; do
        echo "----------------${model_name}"
        echo "------1-----------}"
        CUDA_VISIBLE_DEVICES=0 bash run_vision.sh 1 32 ${model_name} sp ${LOG_DIR} > ${RES_DIR}/image_${model_name}_1.res 2>&1
        sleep 60
        #echo "------8-----------"  # not run
        #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_vision.sh 1 32 ${model_name} sp ${LOG_DIR} > ${RES_DIR}/image_${model_name}_8.res 2>&1
    done
    #------------
    echo "----------------se_resnet50"
        echo "----1----}"
    CUDA_VISIBLE_DEVICES=0 bash run_senet.sh 1 32 se_resnext_50 sp ${LOG_DIR} > ${RES_DIR}/image_se_resnet50_1.res 2>&1
    sleep 60
    #    echo "----8----}"  #not run 
    #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_senet.sh 1 32 se_resnext_50 sp  ${LOG_DIR} > ${RES_DIR}/image_se_resnet50_8.res 2>&1 

}
############################################################################################################################
#######################dy_image_class
dy_image_class(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    
    git clone https://github.com/phlrain/example.git
    pip install torchsummary
    cd ${curl_model_path}/example/image_classification
    mkdir -p data
    ln -s ${datapath}/dygraph_data/ILSVRC2012_Pytorch/ ./data
    ln -s ${datapath}/ILSVRC2012/ ./data
    cp ${BENCHMARK_ROOT}/dynamic_graph/mobilenet/pytorch/run_benchmark.sh ./
    modle_list=(MobileNetV1 MobileNetV2 resnet)
    for model_item in ${modle_list[@]}
    do
        echo "begin to train dynamic image_class(MobileNetV1 MobileNetV2 resnet)--${model_item}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 3 ${model_item} >  ${RES_DIR_DY}/${model_item}.res 2>&1
        sleep 10
    done
}

dy_sequene(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    
    if [ ! -d example ]; then
        git clone https://github.com/phlrain/example.git
    fi
    apt-get install cython
    pip install sacremoses
    pip install --editable .
    pip uninstall sacrebleu -y
    pip install sacrebleu==1.4.0
    cd ${curl_model_path}/example/sequence
    ln -s ${datapath}/dygraph_data/torch_seq2seq_transfomer/ ./data-bin
    cp ${BENCHMARK_ROOT}/dynamic_graph/seq2seq/pytorch/run_benchmark.sh ./run_benchmark_seq2seq.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/transformer/pytorch/run_benchmark.sh ./run_benchmark_transformer.sh
    modle_list=(seq2seq transformer)
    for model_item in ${modle_list[@]}
    do
        echo "begin to train dynamic image_class(seq2seq transformer)--${model_item}"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark_${model_item}.sh 1 sp > ${RES_DIR_DY}/${model_item}.res 2>&1
        sleep 10
    done
}

dy_ptb(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    if [ ! -d example ]; then
        git clone https://github.com/phlrain/example.git
    fi
    cd ${curl_model_path}/example/ptb_lm
    ln -s ${datapath}/dygraph_data/ptb/simple-examples/ ./
    cp ${BENCHMARK_ROOT}/dynamic_graph/ptb/pytorch/run_benchmark.sh ./
    echo "begin to train dynamic ptb"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 3 > ${RES_DIR_DY}/ptb.res 2>&1
}

# pix2pix and cyclegan
dy_gan(){
    curl_model_path=${MODEL_PATH}
    cd ${curl_model_path}
    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

    cd ${curl_model_path}/pytorch-CycleGAN-and-pix2pix
    ln -s ${datapath}/dygraph_data/cityscapes_gan_mini/ ./dataset/cityscapes
    cp ${BENCHMARK_ROOT}/dynamic_graph/gan_models/pytorch/run_benchmark.sh ./
    model_list=(cyclegan pix2pix)
    for model_item in ${model_list[@]}
    do
        echo "begin to train dynamic ${model_item} 1gpu index is speed"
        CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${model_item}  > ${RES_DIR_DY}/${model_item}.res 2>&1
    done
}


run(){
       for model_name in ${cur_model_list[@]}
       do
           begin_timestaps=$(date "+%Y_%m_%d#%H-%M-%S")
           echo "=====================${model_name} run begin==================${begin_timestaps}"
           $model_name
           sleep 60
           end_timestaps=$(date "+%Y_%m_%d#%H-%M-%S")
           echo "*********************${model_name} run end!!******************${end_timestaps}"
       done
}
environment # according to the actual condition
prepare
run

sh ${PYTORCH_BENCHMARK_ROOT}/scripts/py_final_ana.sh ${CUR_DIR}
