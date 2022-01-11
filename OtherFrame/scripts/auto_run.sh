#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

function usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -n  http_proxy  http&https
  -r  root_dir /path/to/OtherFrame/
  -a  all_path /path/to/benchmark_material
  -l  log_path /path/to/logs
  -f  frame  torch|mxnet|tf
  -p  platform local
EOF
}
if [ $# -lt 6 ] ; then
  usage
  exit 1;
fi
while getopts h:n:r:a:l:f:p opt
do
  case $opt in
  h) usage; exit 0 ;;
  n) http_proxy="$OPTARG" ;;
  r) root_dir="$OPTARG" ;;
  a) all_path="$OPTARG" ;;
  l) log_path="$OPTARG" ;;
  f) frame="$OPTARG" ;;
  p) platform="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done


function set_env(){
    export https_proxy=${http_proxy} && export http_proxy=${http_proxy}
    export ROOT_DIR=${root_dir}
    export all_path=${all_path}
    export LOG_PATH_INDEX_DIR=${log_path}/${frame}/index
    export TRAIN_LOG_DIR=${log_path}/${frame}/train_log_dir/
    export RUN_PLAT=${platform:-"local"}
    mkdir -p ${LOG_PATH_INDEX_DIR} ${TRAIN_LOG_DIR}
}



cur_torch_list=(clas_model_torch seg_model_torch speech_model_torch detec_torch_jde-fairmot detec_torch_fast)
cur_mxnet_list=()
cur_tensorflow_list=()

#run_clas_models_torch
clas_model_torch(){
    cur_model_path=${ROOT_DIR}/clas/PyTorch/
    cd ${cur_model_path}
    bash run_Pytorch.sh
}

seg_model_torch(){
    cur_model_path=${ROOT_DIR}/seg/PyTorch
    cd ${cur_model_path}
    echo "------------${cur_model_path}"
    bash run_PyTorch.sh
    cd ${cur_model_path}
    cp *speed ${LOG_PATH_INDEX_DIR}
    cp *1 *8 ${TRAIN_LOG_DIR}
}

speech_model_torch(){
    cur_model_path=${ROOT_DIR}/Speech/PyTorch/PWGAN
    cd ${cur_model_path}
    bash run_PyTorch.sh
    cp scripts/logs/train_log/* ${TRAIN_LOG_DIR}
    cp scripts/logs/index/* ${LOG_PATH_INDEX_DIR}
}

detec_torch_jde-fairmot(){
    cur_model_path=${ROOT_DIR}/detection/PyTorch
    cd ${cur_model_path}
    bash run_PyTorch_mot.sh
    cp models/jde/*.json ${LOG_PATH_INDEX_DIR}
    cp models/jde/*fp32_1    ${TRAIN_LOG_DIR} 
    cp models/jde/*fp32_8    ${TRAIN_LOG_DIR} 
    cp models/fairmot/src/*.json ${LOG_PATH_INDEX_DIR}
    cp models/fairmot/src/*fp32_1    ${TRAIN_LOG_DIR} 
    cp models/fairmot/src/*fp32_8    ${TRAIN_LOG_DIR} 
}

detec_torch_fast(){
    cur_model_path=${ROOT_DIR}/detection/PyTorch
    cd ${cur_model_path} 
    bash run_PyTorch.sh
    cp models/mmdetection/*speed ${LOG_PATH_INDEX_DIR}
    cp models/mmdetection/*fp32_1 ${TRAIN_LOG_DIR}
    cp models/mmdetection/*fp32_8 ${TRAIN_LOG_DIR}
    cp models/mmpose/*speed ${LOG_PATH_INDEX_DIR}
    cp models/mmpose/*fp32_1 ${TRAIN_LOG_DIR}
    cp models/mmpose/*fp32_8 ${TRAIN_LOG_DIR}
    cp models/SOLO/*speed ${LOG_PATH_INDEX_DIR}
    cp models/SOLO/*fp32_1 ${TRAIN_LOG_DIR}
    cp models/SOLO/*fp32_8 ${TRAIN_LOG_DIR}
}

set_env
for model_name in ${cur_torch_list[@]}
    do
        echo "=====================${model_name} run begin=================="
        $model_name
        sleep 60
        echo "*********************${model_name} run end!!******************"
done

