#!/usr/bin/env bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

function LOG {
  echo "[$(basename $0):${BASH_LINENO[0]}]" $* >&2
}

function build_paddle() {
  [ -d Paddle ] || git clone https://github.com/PaddlePaddle/Paddle.git
  pushd Paddle > /dev/null || exit
  git pull
  LOG "[INFO] IMAGE COMMIT ID: $(git log | head -n1 | awk '{print $2}')"
  version=$(date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S")
  branch="develop"
  PADDLE_VERSION=${version}.${branch}.gcc82
  IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
  LOG "[INFO] IMAGE_NAME: ${IMAGE_NAME}"
  [ -f build/python/dist/${IMAGE_NAME} ] && popd >/dev/null && return 0
  env CMAKE_BUILD_TYPE=Release                \
      PY_VERSION=3.7                          \
      PADDLE_VERSION=0.0.0.${PADDLE_VERSION}  \
      WITH_AVX=ON                             \
      WITH_GPU=ON                             \
      WITH_TEST=OFF                           \
      WITH_TESTING=OFF                        \
      RUN_TEST=OFF                            \
      WITH_PYTHON=ON                          \
      WITH_STYLE_CHECK=OFF                    \
      CMAKE_EXPORT_COMPILE_COMMANDS=ON        \
      WITH_MKL=ON                             \
      BUILD_TYPE=Release                      \
      WITH_DISTRIBUTE=OFF                     \
      CMAKE_VERBOSE_MAKEFILE=OFF              \
      /bin/bash paddle/scripts/paddle_build.sh build $(nproc)
  if [ $? -eq 0 -a -f build/python/dist/${IMAGE_NAME} ]
  then
    LOG "[INFO] Build Paddle success!"
  else
    LOG "[INFO] Build Paddle failed!"
    exit -1
  fi
  popd > /dev/null
  mv Paddle/build/python/dist/${IMAGE_NAME} paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
}

function run(){
  [ -f paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl ] || exit -1
  mkdir -p logs
  logs_dir=$(pwd)/logs
  for package in "paddlepaddle" "paddlepaddle-gpu"
  do
    LOG "[INFO] Uninstall $package ..."
    pip uninstall -y $package > /dev/null
  done
  for package in "nvidia-ml-py" "tensorflow==2.3.0" "paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
  do
    LOG "[INFO] Install $package ..."
    pip install $package > /dev/null
  done
  pushd api > /dev/null
  bash deploy/main_control.sh tests_v2 tests_v2/configs ${logs_dir} "${VISIBLE_DEVICES}" "${DEVICES_TYPE}"
  popd > /dev/null
}

[ -f paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl ] || build_paddle
run
