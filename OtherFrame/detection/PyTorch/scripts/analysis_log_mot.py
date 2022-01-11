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

from __future__ import print_function

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    parser.add_argument(
        "--jsonname", type=str, help="The name of dumped json where to output.")
    parser.add_argument(
        "--keyword", type=str, default="time:", help="Keyword to specify analysis data")
    parser.add_argument(
        '--model_name', type=str, default="fairmot", help='training model_name, transformer_base')
    parser.add_argument(
        '--mission_name', type=str, default="目标检测", help='training mission name')
    parser.add_argument(
        '--direction_id', type=int, default=0, help='training direction_id')
    parser.add_argument(
        '--run_mode', type=str, default="sp", help='multi process or single process')
    parser.add_argument(
        '--index', type=int, default=1, help='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    parser.add_argument(
        '--gpu_num', type=int, default=1, help='nums of training gpus')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size of training samples')
    args = parser.parse_args()
    return args


def parse_text_from_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    return lines


def parse_avg_from_text(text: list, keyword: str, skip_line=4):
    count_list = []
    for i, line in enumerate(text):
        if keyword in line:
            words = line.split(" ")
            for j, word in enumerate(words):
                if word == keyword:
                    count_list.append(float(words[j+1][:-1]))
                    break
    count_list = count_list[skip_line:]
    if count_list:
        return round(sum(count_list) / len(count_list), 3)
    else:
        return 0.0


if __name__ == '__main__':
    args = parse_args()
    run_info = dict()
    run_info["log_file"] = args.filename
    res_log_file = args.jsonname
    run_info["model_name"] = args.model_name
    run_info["mission_name"] = args.mission_name
    run_info["direction_id"] = args.direction_id
    run_info["run_mode"] = args.run_mode
    run_info["UNIT"] = "images/s"
    run_info["index"] = args.index
    run_info["gpu_num"] = args.gpu_num
    run_info["FINAL_RESULT"] = 0
    run_info["JOB_FAIL_FLAG"] = 0

    text = parse_text_from_file(args.filename)
    avg_time = parse_avg_from_text(text, args.keyword)
    if avg_time == 0.0:
        run_info["JOB_FAIL_FLAG"] = 1
        print("Failed at get info from training's output log, please check.")
        sys.exit()
    run_info["FINAL_RESULT"] = round(args.batch_size / avg_time * args.gpu_num, 3)

    json_info = json.dumps(run_info)
    with open(res_log_file+'.json', "w") as of:
        of.write(json_info)
