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
    pip install tensorflow-gpu==2.0 pre-commit==1.21 pylint==1.9.5 pytest==4.6.9
    python -c "import tensorflow as tf; print(tf.__version__)"
    apt-get update
    apt-get install -y git
}

function fetch_upstream_master_if_not_exist() {
    UPSTREAM_URL='https://github.com/PaddlePaddle/benchmark'
    origin_upstream_url=`git remote -v | awk '{print $1, $2}' | uniq | grep upstream | awk '{print $2}'` 
    if [ "$origin_upstream_url" == "" ]; then
        git remote add upstream $UPSTREAM_URL.git
    elif [ "$origin_upstream_url" != "$UPSTREAM_URL" ] \
            && [ "$origin_upstream_url" != "$UPSTREAM_URL.git" ]; then
        git remote remove upstream
        git remote add upstream $UPSTREAM_URL.git
    fi
    
    if [ ! -e "$PADDLE_ROOT/.git/refs/remotes/upstream/$BRANCH" ]; then 
        git fetch upstream 
    fi
}

function run_api(){
    fetch_upstream_master_if_not_exist
    cd ${BENCHMARK_ROOT}/api/tests
    HAS_MODIFIED_API_TEST=`git diff --name-only upstream/$BRANCH | grep "api/tests" || true`
    API_NAMES=(abs fc)
    if [ "${HAS_MODIFIED_API_TEST}" != "" ] ; then
        for api in ${HAS_MODIFIED_API_TEST[@]}; do
            new_name=`echo $api |awk -F "/" '{print $NF}' |awk -F "." '{print $NR}'`
            if [[ "$new_name" != "main" && "$new_name" != "feeder" && "$new_name" != "common_ops" ]]; then
                need_append="yes"
                for name in ${API_NAMES[@]}; do
                    if [ "${name}" == "${new_name}" ]; then
                        need_append="no"
                        break
                    fi
                done
                if [ "$need_append" == "yes" ]; then
                    API_NAMES[${#API_NAMES[@]}]=${new_name}
                fi
            fi
        done
    fi

    fail_name=()
    for name in ${API_NAMES[@]}; do
        sh run.sh $name -1
        if [ $? -ne 0 ]; then
            fail_name[${#fail_name[@]}]="$name.py"
        fi
    done
    len=${#fail_name[@]}
    if [ $len -ne 0 ]; then
        echo "Failed API TESTS: ${fail_name[@]}"
        exit 1
    fi
}


function abort(){
    echo "Your change doesn't follow benchmark's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}


function check_style(){
	trap 'abort' 0
	pre-commit install
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

