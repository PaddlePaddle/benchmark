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
  echo [$0:${BASH_LINENO[0]}] $* >&2
}

LOG "[INFO] Start check approval ..."

EXIT_CODE=0
BENCHMARK_ROOT=$(cd $(dirname $0)/../../ && pwd)

declare -A FILE_APPROVAL_USER_MAP
FILE_APPROVAL_USER_MAP=(
  ["api/common/special_op_list.py"]="GaoWei8 wangchaochaohu zhangting2020"
)

LOG "[INFO] Get approval list ..."

declare -A APPROVALED_USER_MAP
approval_line=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/benchmark/pulls/${AGILE_PULL_ID}/reviews?per_page=10000 2> /dev/null)
[ $? -ne 0 ] && LOG "[FATAL] Get review information from github faied" && exit -1
for user in $(echo $approval_line | jq .[].user.login | sed 's|"||g')
do
  APPROVALED_USER_MAP[$user]="approvaled"
  LOG "[INFO] ${user} approval this PR."
done

for file in $(git diff --name-only upstream/master)
do
  if [ -n "${FILE_APPROVAL_USER_MAP[$file]}" ]
  then
    need_approval=1
    approval_user=""
    for user in ${FILE_APPROVAL_USER_MAP[$file]}
    do
      [ -n "${APPROVALED_USER_MAP[$user]}" ] && need_approval=0 && break
      approval_user="$user, ${approval_user}"
    done
    if [ $need_approval -ne 0 ]
    then
      EXIT_CODE=6
      LOG "[FATAL] You must have one RD (${approval_user%,*}) approval for the ${file}"
    fi
  fi
done

[ $EXIT_CODE -ne 0 ] && LOG "[FATAL] Check approval failed." || LOG "[INFO] Check approval success."
exit $EXIT_CODE
