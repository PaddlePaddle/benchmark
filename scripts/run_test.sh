#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

set -ex

if [ -z ${BRANCH} ]; then
    BRANCH="master"
fi

BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." && pwd )"
echo ${BENCHMARK_ROOT}

function prepare_tf_env(){
    pip install tensorflow-gpu==1.15 
}


function run_api(){
    cd ${BENCHMARK_ROOT}/api/tests/
    python abs.py
}

echo "test-bot"
function main(){
    local CMD=$1
    prepare_tf_env
    case $CMD in
      run_api_test)
        run_api 
        ;;
      *)
        echo "Sorry, $CMD not recognized."
        exit 1
        ;;
      esac
      echo "runtest script finished as expected"
}

main $@
