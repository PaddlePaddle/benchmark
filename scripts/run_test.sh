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
    pip install tensorflow-gpu==1.15 pre-commit==1.21 pylint==1.9.5 pytest==4.6.9
    apt-get update
    apt-get install -y git
}


function run_api(){
    cd ${BENCHMARK_ROOT}/api/tests/
    python abs.py
}


function abort(){
    echo "Your change doesn't follow benchmark's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}


function check_style(){
	trap 'abort' 0
	pre-commit run --all-files
	commit_files=on
    	for file_name in `git diff --numstat | awk '{print $NF}'`;do
        	if [ ! pre-commit run --files $file_name ]; then
            		git diff
            		commit_files=off
        	fi
    	done
    	if [ $commit_files == 'off' ];then
        	echo "code format error"
        	exit 1
    	fi
    	trap 0
}


function main(){
    local CMD=$1
    prepare_tf_env
    check_style
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
