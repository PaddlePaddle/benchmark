#!/bin/bash

##########################################
#                                        #
#      usage                             #
#      export the BENCHMARK_ROOT         #
#      export the TF_BENCHMARK_ROOT      #
#                                        #
##########################################

cur_model_list=(deeplabv3 transformer)
#cur_model_list=(padding nextvlad seq2seq deeplabv3 stgan cyclegan transformer)
export https_proxy=http://172.19.56.199:3128
export http_proxy=http://172.19.56.199:3128
##   config.gpu_options.allow_growth = True

######################
environment(){
apt-get update
apt-get install wget -y 
apt-get install vim -y
apt-get install tk-dev -y
apt-get install python-tk -y
#pip uninstall tensorflow-gpu -y
#pip install tensorflow-gpu==1.15.0
package_check_list=(pytest Cython opencv-python future pycocotools matplotlib networkx fasttext visdom  Pillow)
    for package in ${package_check_list[@]}; do
        if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
        else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
        fi
done
}


#################pip packages
prepare(){
export BENCHMARK_ROOT=/ssd3/heya/tensorflow/benchmark_push/benchmark/
export TF_BENCHMARK_ROOT=${BENCHMARK_ROOT}/competitive_products/tf_benchmark

export datapath=/ssd1/ljh/dataset

cur_timestaps=$(date "+%Y%m%d%H%M%S")
export CUR_DIR=${TF_BENCHMARK_ROOT}/${cur_timestaps}_result/
export LOG_DIR=${CUR_DIR}/LOG
export RES_DIR=${CUR_DIR}/RES
export MODEL_PATH=${CUR_DIR}/tf_models

mkdir -p ${LOG_DIR}
mkdir -p ${RES_DIR}
mkdir -p ${MODEL_PATH}
}

#########padding
padding(){
killall -9 nvidia-smi
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

cp -r ${BENCHMARK_ROOT}/static_graph/PaddingRNN/lstm_tf ${curl_model_path}/padding
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_padding.sh ${curl_model_path}/padding
cd ${curl_model_path}/padding

rm -rf data
mkdir data
ln -s ${datapath}/simple-examples/ ./data/

model_type_list=(large small)
rnn_type_list=(static padding)
for model_type in ${model_type_list[@]}; do
    for rnn_type in ${rnn_type_list[@]}; do
        model_name="padding_${model_type}_${rnn_type}"
        echo "-----------------------$model_name begin!"
        CUDA_VISIBLE_DEVICES=0 bash run_padding.sh speed ${model_type} ${rnn_type} ${LOG_DIR} > ${RES_DIR}/speed_${model_name}_1.res 2>&1
        echo "$model_name speed finished!"
        sleep 60
        CUDA_VISIBLE_DEVICES=0 bash run_padding.sh mem ${model_type} ${rnn_type} ${LOG_DIR} > ${RES_DIR}/mem_${model_name}_1.res 2>&1
        echo "$model_name mem finished!"
        sleep 60
     done
done
}

###################3nextvlad
nextvlad(){
# NOTE: make sure you rm nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic/ when you can not address a error
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

git clone https://github.com/linrongc/youtube-8m.git
cd ${curl_model_path}/youtube-8m/

mkdir -p ${curl_model_path}/youtube-8m/data
ln -s ${datapath}/yt8m/ ${curl_model_path}/youtube-8m/data/
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_nextvlad.sh ${curl_model_path}/youtube-8m/

run_type_list=(speed mem)  
for run_type in ${run_type_list[@]};do
    CUDA_VISIBLE_DEVICES=0 bash run_nextvlad.sh train ${run_type} sp  ${LOG_DIR} > ${RES_DIR}/${run_type}_nextvlad_1.res 2>&1
    echo "-------------------1cards $run_type end"
    sleep 60
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_nextvlad.sh train ${run_type} sp  ${LOG_DIR} > ${RES_DIR}/${run_type}_nextvlad_8.res 2>&1
    echo "-------------------8cards $run_type end"
    sleep 60
done
}

#####################seq2seq
seq2seq(){
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

cp -r ${BENCHMARK_ROOT}/static_graph/seq2seq/tensorflow/ ${curl_model_path}/seq2seq
cd ${curl_model_path}/seq2seq

ln -s ${datapath}/tf_seq2seq_data/ ${curl_model_path}/seq2seq/nmt/data
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_seq2seq.sh ${curl_model_path}/seq2seq


run_type_list=(speed mem) 
for run_type in ${run_type_list[@]};do
    CUDA_VISIBLE_DEVICES=0 bash run_seq2seq.sh ${run_type} sp  ${LOG_DIR} > ${RES_DIR}/${run_type}_seq2seq_1.res 2>&1
    echo "------------------- $run_type end"
    sleep 60
done
}
#################33deepv3
deeplabv3(){
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

mkdir -p ${curl_model_path}/deeplabv3
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_deeplabv3.sh ${curl_model_path}/deeplabv3
cd ${curl_model_path}/deeplabv3

run_type_list=(speed mem)  
for run_type in ${run_type_list[@]};do
    echo "---------------$run_type----"
    CUDA_VISIBLE_DEVICES=0 bash run_deeplabv3.sh ${run_type} sp ${LOG_DIR} > ${RES_DIR}/${run_type}_deeplab_1.res 2>&1 
    echo "---------1card finished"
    sleep 60
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_deeplabv3.sh ${run_type} sp ${LOG_DIR} > ${RES_DIR}/${run_type}_deeplab_8.res 2>&1
    echo "----------8card finished"
    sleep 60
done
}
########################stgan
stgan(){
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

git clone  https://github.com/csmliu/STGAN.git
cd ${curl_model_path}/STGAN

cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_stgan.sh ${curl_model_path}/STGAN
mkdir -p ${curl_model_path}/STGAN/data/celeba
ln -s ${datapath}/CelebA/Img/img_align_celeba ${curl_model_path}/STGAN/data/celeba/img_align_celeba
ln -s ${datapath}/CelebA/Anno/list_attr_celeba.txt ${curl_model_path}/STGAN/data/celeba/list_attr_celeba.txt


run_type_list=(speed mem)  
for run_type in ${run_type_list[@]};do
    echo "---------------$run_type----"
    CUDA_VISIBLE_DEVICES=0 bash run_stgan.sh train ${run_type} ${LOG_DIR} > ${RES_DIR}/${run_type}_stgan_1.res 2>&1 
    sleep 60
done

}
##################cyclegan
cyclegan(){
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

cp -r ${BENCHMARK_ROOT}/static_graph/CycleGAN/tensorflow ${curl_model_path}/cyclegan
cd ${curl_model_path}/cyclegan

mkdir -p ${curl_model_path}/cyclegan/input
ln -s ${datapath}/horse2zebra/  ${curl_model_path}/cyclegan/input/
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_cyclegan.sh ${curl_model_path}/cyclegan/

run_type_list=(speed mem)  
for run_type in ${run_type_list[@]};do
    echo "---------------$run_type----"
    CUDA_VISIBLE_DEVICES=0 bash run_cyclegan.sh ${run_type} ${LOG_DIR} > ${RES_DIR}/${run_type}_cyclegan_1.res 2>&1
    sleep 60 
done
}


transformer(){
pip uninstall tensor2tensor tensorflow-gpu -y
pip install tensor2tensor==1.9.0
pip install tensorflow-gpu==1.15.0

sed -i '67c \#from tensor2tensor.models.video import savp'   /usr/local/lib/python2.7/dist-packages/tensor2tensor/models/__init__.py   # for run
sed -i '81 s/^/  config.gpu_options.allow_growth = False\n/' /usr/local/lib/python2.7/dist-packages/tensor2tensor/utils/trainer_lib.py   # for mem, this is set in run_transformer.sh

curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

cp -r ${BENCHMARK_ROOT}/static_graph/Transformer/tensor2tensor ${curl_model_path}/transformer
cd ${curl_model_path}/transformer

ln -s ${datapath}/transformer/data ${curl_model_path}/transformer/
cp ${TF_BENCHMARK_ROOT}/scripts/run_scripts/run_transformer.sh ${curl_model_path}/transformer

run_type_list=(speed mem)
model_type_list=(big base)
for run_type in ${run_type_list[@]}; do
    for model_type in ${model_type_list[@]}; do
    model_name=transfomer_${model_type}
    CUDA_VISIBLE_DEVICES=0 bash run_transformer.sh ${run_type} ${model_type} ${LOG_DIR} >  ${RES_DIR}/${run_type}_${model_name}_1.res 2>&1
    echo "-----${model_name} 1card ${run_type} finished"
    sleep 60
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_transformer.sh ${run_type} ${model_type} ${LOG_DIR} >  ${RES_DIR}/${run_type}_${model_name}_8.res 2>&1
    echo "-----${model_name} 8card ${run_type} finished"
    sleep 60
    done
done

echo "`pip list | grep "tensorflow-gpu"`"
pip uninstall tensorflow-gpu -y
pip install tensorflow-gpu==1.15.0
}




run(){
       for model_name in ${cur_model_list[@]}
       do
           echo "=====================${model_name} run begin=================="
           $model_name
           sleep 60
           echo "*********************${model_name} run end!!******************"
       done
}
environment  # According to actual situation
prepare
run

sh ${TF_BENCHMARK_ROOT}/scripts/tf_final_ana.sh ${CUR_DIR}
