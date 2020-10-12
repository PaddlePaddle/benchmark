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

set +ex

BENCHMARK_ROOT=$(cd $(dirname $0)/../.. && pwd)

declare -A OP_FILE_MAP
declare -A OP_CONFIG_MAP
declare -A OP_NO_BACKWARD_MAP

declare -A DEVICE_TASK_PID_MAP
declare -A TASK_PID_INFO_MAP

function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}

function prepare_env() {
  LOG "[INFO] Update pip ..."
  python -m pip install --upgrade pip > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Update pip failed!" && exit -1
  
  PADDLE_WHL="paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
  PADDLE_URL="https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/${PADDLE_WHL}"
  LOG "[INFO] Downloading paddle wheel from ${PADDLE_URL}, this could take a few minutes ..."
  wget -q -N -O ${PADDLE_WHL} ${PADDLE_URL}
  [ $? -ne 0 ] && LOG "[FATAL] Download paddle wheel failed!" && exit -1
  LOG "[INFO] Installing paddle, this could take a few minutes ..."
  pip install -U ${PADDLE_WHL} > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Install paddle failed!" && exit -1

  for package in "tensorflow==2.3.0" "tensorflow-probability" "pytest==4.6.9"
  do
    LOG "[INFO] Installing $package, this could take a few minutes ..."
    pip install $package > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Install $package failed!" && exit -1
  done
  python -c "import tensorflow as tf; print(tf.__version__)" > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Install tensorflow success, but it can't work!" && exit -1

  apt-get update > /dev/null 2> /dev/null
  apt-get install -y jq > /dev/null 2> /dev/null

  [ -d /output-logs ] && rm -rf /output-logs/* || mkdir /output-logs
}


function init_device_maps() {
  for device_id in {0..7}
  do
    DEVICE_TASK_PID_MAP[$device_id]=0
  done
}

function load_op_maps() {
  LOG "[INFO] Load all OPs ..."
  files=$(find ${BENCHMARK_ROOT}/api/tests* -type f -name "*.py")
  files=$(echo "$files" | grep -oP ".*tests_v2/(?!main|__init__|common_import|launch).*")
  for file in $files
  do
    file_name=$(echo $file | grep -oP "[^/]*\.py" | grep -oP "^[^\.]+")
    file_content=$(cat $file)
    alias_config_line=$(echo "${file_content}" | grep -oP "self.alias_config ?= ?.*$")
    if [ -n "$alias_config_line" ]
    then
      config=$(echo $alias_config_line | grep -oP '(".*")|([^ ]*Config\(\))')
      config=${config%Config()*} && config=${config,,} && config=${config//\"/}
    else
      config=$file_name
    fi
    api_list_line=$(echo ${file_content} | grep -oP "self.api_list ?= ?{[^}]*}")
    if [ -n "$api_list_line" ]
    then
      api_list=$(echo $api_list_line | tr "'" '"' | grep -oP "{.*}" | jq -r ". | keys[]")
    else
      api_list=$file_name
    fi
    for api in $api_list
    do
      LOG "[INFO] From file ${file} load OP = ${api}, Config = ${config}"
      OP_FILE_MAP[$api]=$file
      OP_CONFIG_MAP[$api]=$config
    done
  done
}

function load_no_backward_ops() {
  LOG "[INFO] Load no backward OPs ..."
  file_content=$(cat ${BENCHMARK_ROOT}/api/common/special_op_list.py)
  no_backward_ops_line=$(echo $file_content | grep -oP "NO_BACKWARD_OPS ?= ?\[[^\]]*\]")
  for op in $(echo $no_backward_ops_line | grep -oP "\[.*\]" | jq -r .[])
  do
    OP_NO_BACKWARD_MAP[$op]="true"
  done
  LOG "[INFO] Load no backward ops: ${!OP_NO_BACKWARD_MAP[*]}"
}

function print_detail_status() {
  [ $1 -eq 0 ] && return
  task_pid=$1
  wait $task_pid
  exit_code=$?
  if [ $exit_code -eq 0 ]
  then
    exit_status="SUCCESS"
  elif [ $exit_code -eq 124 ]
  then
    exit_status="TIMEOUT"
  else
    exit_status="FAILED"
  fi
  LOG "[INFO] ${TASK_PID_INFO_MAP[$task_pid]} ***${exit_status}***"
}

function get_one_free_id() {
  while true
  do
    for device_id in ${!DEVICE_TASK_PID_MAP[*]}
    do
      task_pid=${DEVICE_TASK_PID_MAP[$device_id]}
      if [ $task_pid -eq 0 -o -z "$(ps -opid | grep $task_pid)" ]
      then
        echo $device_id && return 0
      fi
    done
    sleep 1s
  done
}

function run_all_ops() {
  pushd $BENCHMARK_ROOT/api > /dev/null
  total_size=${#OP_FILE_MAP[*]}
  for place in "GPU" "CPU"
  do
    op_index=1
    [ "${place}" == "GPU" ] && repeat=1000 || repeat=100
    [ "${place}" == "GPU" ] && use_gpu=True || use_gpu=False
    for op in ${!OP_FILE_MAP[*]}
    do
      case_file=$BENCHMARK_ROOT/api/tests_v2/configs/${OP_CONFIG_MAP[$op]}.json
      [ ! -f "$case_file" ] && LOG "[FATAL] There is no file $case_file" && continue
      case_size=$(cat $case_file | jq ". | length")
      for case_index in $(seq 1 $case_size)
      do
        for task in "speed" "accuracy"
        do
          for framework in "paddle" "tensorflow"
          do
            [ "accuracy" == "$task" -a "tensorflow" == "$framework" ] && continue
            for direction in "forward" "backward"
            do
              [ "backward" == "$direction" -a "true" == "${OP_NO_BACKWARD_MAP[$op]}" ] && continue
              [ "backward" == "$direction" ] && backward="True" || backward="False"
              device_id=$(get_one_free_id)
              print_detail_status ${DEVICE_TASK_PID_MAP[$device_id]}
              task_info="[Place:${place}] [Device Id:${device_id}]
                         [OP:${op}($op_index/$total_size)] [Case:$case_index/$case_size]
                         [Task:${task}] [Framework:${framework}] [Direction:${direction}]"
              LOG "[INFO] ${task_info} start ..."
              if [ "${place}" == "GPU" ]
              then
                device_limit="env CUDA_VISIBLE_DEVICES=${device_id}"
              else
                device_limit="env CUDA_VISIBLE_DEVICES= taskset -c ${device_id}"
              fi
              command="${device_limit} timeout 600s
                       python -m common.launch ${OP_FILE_MAP[$op]}
                              --api_name ${op}
                              --task ${task}
                              --framework ${framework}
                              --json_file ${case_file}
                              --config_id $((case_index - 1))
                              --backward ${backward}
                              --use_gpu ${use_gpu}
                              --repeat ${repeat}
                              --allow_adaptive_repeat True"
              log_file=/output-logs/${op}_$((case_index - 1))-${framework}_${place}_${task}_${direction}.txt
              ${command} > ${log_file} 2>&1 &
              task_pid=$!
              DEVICE_TASK_PID_MAP[$device_id]=$task_pid
              TASK_PID_INFO_MAP[$task_pid]=${task_info}
            done
          done
        done
      done
      op_index=$((op_index + 1))
    done
  done
  popd > /dev/null
  for task_pid in ${DEVICE_TASK_PID_MAP[*]}
  do
    print_detail_status $task_pid
  done
}

function main() {
  LOG "[INFO] Start run daily op benchmark test ..."
  prepare_env
  init_device_maps
  load_op_maps
  load_no_backward_ops
  run_all_ops
  LOG "[INFO] Daily op benchmark run success and no error!"
}

main
