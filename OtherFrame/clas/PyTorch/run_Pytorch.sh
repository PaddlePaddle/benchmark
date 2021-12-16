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

root_dir=${ROOT_DIR:-"/workspace"}                          # /path/to/clas
all_path=${all_path}                                        # /path/to/dataset&whls
log_path_index_dir=${LOG_PATH_INDEX_DIR:-$(pwd)}            # /path/to/result
train_log_dir=${TRAIN_LOG_DIR:-$(pwd)}                      # /path/to/logs
run_plat=${RUN_PLAT:-"local"}                               # wheter downloading dataset

run_cmd="cd /workspace;
         bash test_clas.sh"

# 拉镜像
ImageName="paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82";
docker pull ${ImageName}

# 启动镜像
nvidia-docker run  -i --rm  \
    --name test_pytorch_clas \
    --net=host \
    --cap-add=ALL \
    --shm-size=64g \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e ROOT_DIR=${root_dir} \
    -e LOG_PATH_INDEX_DIR=${log_path_index_dir} \
    -e TRAIN_LOG_DIR=${train_log_dir} \
    -e RUN_PLAT=${run_plat} \
    -e all_path=${all_path} \
    -v $PWD:/workspace \
    -v /ssd3:/ssd3 \
    -v /ssd2:/ssd2 \
    ${ImageName}  \
    /bin/bash -c "${run_cmd}"
nvidia-docker stop test_pytorch_clas
nvidia-docker rm test_pytorch_clas

