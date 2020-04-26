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
    apt-get update
    apt-get -y install git
    pip install tensorflow-gpu==1.15 
}

function fetch_upstream_develop_if_not_exist() {
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

function generate_upstream_master_api_spec() {
    fetch_upstream_develop_if_not_exist
    cur_branch=`git branch | grep \* | cut -d ' ' -f2`
    git checkout -b master_base_pr upstream/$BRANCH
    generate_api_spec "DEV"
    git branch -D master_base_pr
}

function generate_api_spec() {
    spec_kind=$1
    if [ "$spec_kind" != "PR" ] && [ "$spec_kind" != "DEV" ]; then
        echo "Not supported $2"
        exit 1
    fi
    spec_path=${BENCHMARK_ROOT}/scripts/API_${spec_kind}.spec
    cd ${BENCHMARK_ROOT}
    python scripts/print_signature.py api/tests > $spec_path
}

function run_api(){
    cd ${BENCHMARK_ROOT}
    python scripts/diff_api.py
}


function main(){
    local CMD=$1
    prepare_tf_env
    case $CMD in
      run_api_test)
        generate_upstream_master_api_spec
        generate_api_spec "PR"
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
