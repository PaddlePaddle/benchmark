#!/bin/bash
##########################################
#                                        #
#      usage                             #
#      export the BENCHMARK_ROOT         #
#      export the MX_BENCHMARK_ROOT      #
#                                        #
##########################################

cur_model_list=(yolov3)

environment(){
export LD_LIBRARY_PATH=/home/work/418.39/lib64/:/usr/local/cuda-10.0/compat/:$LD_LIBRARY_PATH
apt-get update
apt-get install vim git psmisc python-tk yum -y

check_package_list=(gluoncv Cython)
for package in ${check_package_list[@]};
do
      if python -c "import ${package}" >/dev/null 2>&1; then
            echo "${package} have already installed"
      else
            echo "${package} NOT FOUND"
            pip install ${package}
            echo "${package} installed"
      fi
done

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install --user

cd /usr/bin/
ln -s /ssd2/liyang/tools/monquery
ln -s /ssd2/liyang/tools/monqueryfunctions
cd -

}

prepare(){
export BENCHMARK_ROOT=/ssd3/heya/mxnet/0619_code/benchmark/
export MX_BENCHMARK_ROOT=${BENCHMARK_ROOT}/competitive_products/mxnet_benchmark

export datapath=/ssd1/ljh/dataset

cur_timestaps=$(date "+%Y%m%d%H%M%S")
export CUR_DIR=${MX_BENCHMARK_ROOT}/${cur_timestaps}_result/
export LOG_DIR=${CUR_DIR}/LOG
export RES_DIR=${CUR_DIR}/RES
export MODEL_PATH=${CUR_DIR}/mx_models

mkdir -p ${LOG_DIR}
mkdir -p ${RES_DIR}
mkdir -p ${MODEL_PATH}

}

yolov3(){
curl_model_path=${MODEL_PATH}
cd ${curl_model_path}

cp -r ${BENCHMARK_ROOT}/static_graph/yolov3/gluon_cv ${curl_model_path}/yolov3
cd ${curl_model_path}/yolov3/yolo
cp ${BENCHMARK_ROOT}/static_graph/yolov3/mxnet/run.sh ${curl_model_path}/yolov3/yolo/run_yolov3.sh

mkdir -p ~/.mxnet/datasets
ln -s ${datapath}/COCO17/ ~/.mxnet/datasets/coco 

echo "----run one card----"
CUDA_VISIBLE_DEVICES=0 bash run_yolov3.sh train 1 ${LOG_DIR} > ${RES_DIR}/yolov3_1.res 2>&1 
sleep 60
echo "----run 8 card----"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_yolov3.sh train 1 ${LOG_DIR} > ${RES_DIR}/yolov3_8.res 2>&1 

}

run(){
for model_name in ${cur_model_list[@]}; do
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

sh ${MX_BENCHMARK_ROOT}/scripts/mx_final_ana.sh ${CUR_DIR}
