#run resnet
ResNet50_bs32_dygraph(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    git checkout 98db91b2118deb0f6f1c0bf90708c1bc34687f8d
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark_dygraph.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./run_benchmark_dygraph.sh
    sed -i '/set\ -xe/d' run_benchmark_dygraph.sh

    #running models cases
    model_name=ResNet50_bs32_dygraph
    run_batchsize=32
    echo "index is speed, 2gpu, begin, ResNet50_bs32_dygraph"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_dygraph.sh 1 ${run_batchsize} ${model_name} mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
}

ResNet50_bs32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
    git checkout a8f21e0167e4de101cbcd241b575fb09bbcaced9
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark_static.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark_resnet.sh ./run_benchmark_static.sh
    sed -i '/set\ -xe/d' run_benchmark_static.sh
    #running models cases
    model_name=ResNet50_bs32
    run_batchsize=32
    echo "index is speed, 2gpu, begin, ResNet50_bs32_static"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark_static.sh 1 ${run_batchsize} ${model_name} mp 1 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat ${model_name}_1_2_mp
}


#run bert_base_fp32
bert_base_seqlen128_fp32_bs32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/language_model/bert/
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en_seqlen512 ${cur_model_path}/wikicorpus_en_seqlen512 
    mv wikicorpus_en_seqlen512 ./data
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ${cur_model_path}/wikicorpus_en_seqlen128
    mv wikicorpus_en_seqlen128 ./data
    rm -rf /root/.paddlenlp/models
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/bert/paddle/run_benchmark.sh ./run_benchmark.sh
    pip install paddlenlp
    python -c 'import paddlenlp'  #to make dir /root/.paddlenlp/models before model running 

    sed -i '/set\ -xe/d' run_benchmark.sh
    model_mode=base
    fp_mode=fp32
    seq_item=seqlen128
    bs_item=32
    model_name="bert_${model_mode}_${seq_item}_${fp_mode}_bs${bs_item}"
    echo "index is speed, 2gpus, begin, ${model_name}"
    export FLAGS_call_stack_level=2
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 ${model_mode} ${fp_mode} mp ${bs_item} 400  ${seq_item} | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
    unset FLAGS_call_stack_level    
}
#transformer
transformer_base_bs4096_amp_fp16(){
    pip install paddlenlp==2.0.5
    pip install attrdict
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/machine_translation/transformer
    cd ${cur_model_path}
    #prepare data
    mkdir -p ~/.paddlenlp/datasets
    ln -s ${data_path}/transformer/WMT14ende ~/.paddlenlp/datasets/ 
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/transformer/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    mode_item=base
    fp_item=amp_fp16
    bs=4096
    model_name="transformer_${mode_item}_bs${bs}_${fp_item}"
    echo "index is speed, ${model_name} 2gpu begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 mp 500 ${mode_item} ${fp_item} | tee  ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_2gpus 2>&1   
    sleep 1s
    cat dynamic_${model_name}_1_2_mp
}
#yolov3
yolov3_bs8(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    git branch    #develop 分支
    cd ${cur_model_path}
    pip install Cython
    pip install pycocotools
    pip install -r requirements.txt 
   
    mkdir -p ~/.cache/paddle/weights
    ln -s ${prepare_path}/yolov3/DarkNet53_pretrained ~/.cache/paddle/weights
    cd ${cur_model_path}
    echo "-------before data prepare"
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ./dataset/coco
    echo "-------after data prepare"
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/yolov3/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 2gpu, begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 mp 500 | tee ${BENCHMARK_ROOT}/logs/dynamic/yolov3_bs1_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_yolov3_1_2_mp
}
#tsm
TSM_bs16(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleVideo
    cd ${cur_model_path}

    pip install wget decord
    # Prepare pretrained modles
    ln -s ${prepare_path}/tsm/ResNet50_pretrain.pdparams ${cur_model_path}/
    # Prepare data
    rm -rf ./data/ucf101
    ln -s ${data_path}/TSM/ucf101_Vedio/ ${cur_model_path}/data/ucf101

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/tsm/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 2gpu begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/tsm_bs_16_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_TSM_1_2_mp
}
#deeplabv3
deeplabv3_bs4(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleSeg/
    cd ${cur_model_path}
    pip install  visualdl
    # Prepare data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/cityscapes_hrnet_torch ${cur_model_path}/data/cityscapes

    # Running
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seg_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh

    model_item=deeplabv3 
    bs_item=4
    echo "index is speed, ${model_item} 2gpu begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh 1 ${bs_item} mp ${model_item} 200 | tee ${BENCHMARK_ROOT}/logs/dynamic/seg_${model_item}_bs${bs_item}_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_deeplabv3_1_2_mp
}

#run CycleGAN
CycleGAN_bs1(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleGAN
    cd ${cur_model_path}

    if python -c "import pooch" >/dev/null 2>&1; then
        echo "pooch have already installed, need uninstall"
        pip uninstall -y pooch
    else
        echo "pooch not installed"
    fi

    pip install -r requirements.txt
    pip install scikit-image==0.18.1
    pip install easydict
    # Prepare data
    mkdir -p data
    ln -s ${data_path}/cityscapes_gan_mini ${cur_model_path}/data/cityscapes

    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/gan_models/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    model_item=CycleGAN
    echo "index is speed, ${model_item} 1gpu begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp ${model_item} 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/gan_${model_item}_bs1_speed_1gpus 2>&1
    sleep 1s
    cat dynamic_CycleGAN_bs1_1_1_sp
}
#mask_rcnn
mask_rcnn_bs1(){
#ResNet50_bs32_dygraph
    cur_model_path=${BENCHMARK_ROOT}/PaddleDetection
    cd ${cur_model_path}
    pip install Cython
    pip install pycocotools
    pip install -r requirements.txt 


    package_check_list=(imageio tqdm Cython pycocotools tb_paddle scipy)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
        else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
        fi
    done

    # Copy pretrained model
    ln -s ${prepare_path}/mask-rcnn/ResNet50_cos_pretrained  ~/.cache/paddle/weights
    cd ${cur_model_path}
    # Prepare data
    rm -rf dataset/coco
    ln -s ${data_path}/COCO17 ${cur_model_path}/dataset/coco
    # preprare scripts
    rm -rf run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mask_rcnn/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 2gpu begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 500 | tee ${BENCHMARK_ROOT}/logs/dynamic/mask_rcnn_bs1_speed_2gpus 2>&1   
    sleep 1s
    cat dynamic_mask_rcnn_1_2_mp    
}

PPOCR_mobile_2_bs8(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleOCR
    cd ${cur_model_path}
    pip install -r requirements.txt

    if python -c "import pooch" >/dev/null 2>&1; then
        echo "pooch have already installed, need uninstall"
        pip uninstall -y pooch
    else
        echo "pooch not installed"
    fi

    package_check_list=(shapely scikit-image imgaug pyclipper lmdb tqdm numpy visualdl python-Levenshtein)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
        else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
        fi
    done
    # Prepare data
    rm -rf train_data/icdar2015
    if [ ! -d "train_data" ]; then
        mkdir train_data
    fi
    ln -s ${data_path}/PPOCR_mobile_2.0/icdar2015 ${cur_model_path}/train_data/icdar2015
    rm -rf pretrain_models
    ln -s ${prepare_path}/PPOCR_mobile_2.0/pretrain_models ./pretrain_models
    
    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/ppocr_mobile_2/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 2gpu begin"
    CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh  1 mp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/ppocr_mobile_2_bs8_speed_2gpus 2>&1
    sleep 1s
    cat dynamic_PPOCR_mobile_2_bs8_1_2_mp      
}

seq2seq_bs128(){
    cur_model_path=${BENCHMARK_ROOT}/models/dygraph/seq2seq
    cd ${cur_model_path}
    # Prepare data
    ln -s ${data_path}/seq2seq/data/ ${cur_model_path}/data
    # Running ...
    rm -f ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/seq2seq/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    echo "index is speed, 1gpu begin"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/seq2seq_bs128_speed_1gpus 2>&1
    sleep 1s
    cat dynamic_seq2seq_bs128_1_1_sp   
}
