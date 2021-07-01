#run resnet
ResNet50_bs32_dygraph(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    git checkout -b develop_resnet 98db91b2118deb0f6f1c0bf90708c1bc34687f8d
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark_dygraph.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./run_benchmark_dygraph.sh
    sed -i '/set\ -xe/d' run_benchmark_dygraph.sh

    #running models cases
    model_name=ResNet50_bs32_dygraph
    run_batchsize=32
    echo "index is speed, 1gpu, begin, ResNet50_bs32_dygraph"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark_dygraph.sh 1 ${run_batchsize} ${model_name} sp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_1gpus 2>&1
    cat dynamic_${model_name}_1_1_sp
    sleep 60
}

ResNet50_bs32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    git checkout -b static a8f21e0167e4de101cbcd241b575fb09bbcaced9
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark_static.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark_resnet.sh ./run_benchmark_static.sh
    sed -i '/set\ -xe/d' run_benchmark_static.sh
    #running models cases
    model_name=ResNet50_bs32
    run_batchsize=32
    echo "index is speed, 1gpu, begin, ResNet50_bs32_static"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark_static.sh 1 ${run_batchsize} ${model_name} sp 1 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_1gpus 2>&1
    cat ${model_name}_1_1_sp
    sleep 60
}


#run bert_base_fp32
bert_base_seqlen128_fp32_bs32(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/examples/language_model/bert/static
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ${cur_model_path}/wikicorpus_en_seqlen128
    rm -rf ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/BERT/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    
    #running model case
    model_name=bert_base_seqlen128_fp32_bs32
    echo "index is speed, 1gpu, begin, bert_base_fp32"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 base fp32 sp 32 500 seqlen128 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_1gpus 2>&1
    cat ${model_name}_1_1_sp
    sleep 60    
}

#run MobileNetV1
MobileNetV1(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    git checkout -b develop_mobilenet 98db91b2118deb0f6f1c0bf90708c1bc34687f8d
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -f ./run_benchmark_mobilenet.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/mobilenet/paddle/run_benchmark_mobilenet.sh ./
    sed -i '/set\ -xe/d' run_benchmark_mobilenet.sh

    #running model case
    model_name=MobileNetV1
    echo "index is speed, 1gpu, begin, MobileNetV1"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark_mobilenet.sh 1  sp 1  ${model_name} | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_1gpus 2>&1
    sleep 60
}

#run CycleGAN
CycleGAN(){
    cur_model_path=${BENCHMARK_ROOT}/models/PaddleCV/gan
    cd ${cur_model_path}
    #prepare data
    mkdir -p ${cur_model_path}/data
    ln -s ${data_path}/horse2zebra/ ${cur_model_path}/data/cityscapes
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/CycleGAN/paddle/run_benchmark.sh ./
    sed -i '/set\ -xe/d' run_benchmark.sh
    
    #running model case
    model_name=CycleGAN
    echo "index is speed, begin, CycleGAN"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 sp 600 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_1gpus 2>&1
    sleep 60
}
#ResNet50_bs32_dygraph
#ResNet50_bs32
#bert_base_seqlen128_fp32_bs32
#MobileNetV1
#CycleGAN
