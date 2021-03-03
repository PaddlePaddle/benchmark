resnet(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
    # Prepare data
    ln -s ${data_path}/imagenet100_data/ ${cur_model_path}/dataset
    # Copy run_benchmark.sh and running ...
    rm -rf ./run_benchmark_static.sh
    rm -rf ./run_benchmark_dygraph.sh
    cp ${BENCHMARK_ROOT}/static_graph/image_classification/paddle/run_benchmark_resnet.sh ./run_benchmark_static.sh
    cp ${BENCHMARK_ROOT}/dynamic_graph/resnet/paddle/run_benchmark_resnet.sh ./run_benchmark_dygraph.sh
    sed -i '/set\ -xe/d' run_benchmark_static.sh
    sed -i '/set\ -xe/d' run_benchmark_dygraph.sh

    #running models cases
    model_name=ResNet50_bs32
    run_batchsize=32
    echo "index is speed, 1gpu, begin, ResNet50_bs32_static"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark_static.sh 1 ${run_batchsize} ${model_name} sp 1 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_1gpus 2>&1
    sleep 60
    echo "index is speed, 1gpu, begin, ResNet50_bs32_dygraph"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark_dygraph.sh 1 ${run_batchsize} ${model_name} sp 1 | tee ${BENCHMARK_ROOT}/logs/dynamic/${model_name}_speed_1gpus 2>&1
    sleep 60
}
#run bert_base_fp32
bert(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleNLP/benchmark/bert
    cd ${cur_model_path}
    ln -s ${data_path}/Bert/wikicorpus_en_seqlen128 ${cur_model_path}/wikicorpus_en_seqlen128
    rm -rf ./run_benchmark.sh
    cp ${BENCHMARK_ROOT}/static_graph/BERT/paddle/run_benchmark.sh ./run_benchmark.sh
    sed -i '/set\ -xe/d' run_benchmark.sh
    
    #running model case
    model_name=bert_base_seqlen128_fp32_bs32
    echo "index is speed, 1gpu, begin, bert_base_fp32"
    CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh 1 base fp32 sp 500 | tee ${BENCHMARK_ROOT}/logs/static/${model_name}_speed_1gpus 2>&1
    sleep 60    
}
#run MobileNetV1
mobilenetV1(){
    cur_model_path=${BENCHMARK_ROOT}/PaddleClas
    cd ${cur_model_path}
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
resnet
bert
mobilenetV1
