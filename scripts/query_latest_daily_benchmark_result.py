#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# 目前仅适用于 V100 && CUDA 10.1 版本。
# 对于之前已有的数据，从logs目录下直接拷贝过来
# 对于之前没有的数据，写入一个 json 文件中等待后续解析并执行
# 接收参数
# 1. dynamic_models:
# 2. static_models:
# 3. save_dir:
# 4. output_file:

import os
import copy
import json
import argparse
import shutil
import requests


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--static_models",
    type=str,
    default='',
    help="query static models list")
parser.add_argument(
    "--dynamic_models",
    type=str,
    default='',
    help="query dynamic models list")
parser.add_argument(
    "--save_dir",
    type=str,
    default='./diff',
    help="index result save dir")
parser.add_argument(
    "--output_file",
    type=str,
    default='.rerun_list',
    help="output_file, record which model need to rerun")


def _query_latest_image_id():
    """
    # 调用 HTTP 接口查询最新的 image_id
    """
    url = "http://yq01-page-powerbang-table1077.yq01.baidu.com:8988/benchmark/image/"
    params = {
        "cuda_version": "10.1",
        "image_type": "2"
    }
    static_response = requests.get(url, params=params).json()
    static_image_info = None
    dynamic_image_info = None
    for item in static_response:
        static_image_info = {
            "image_id": item["image_id"],
            "version": item["version"],
            "image_commit_id": item["image_commit_id"]
        }
        break
    params = {
        "cuda_version": "10.1",
        "image_type": "3"
    }
    dynamic_response = requests.get(url, params=params).json()
    for item in dynamic_response:
        dynamic_image_info = {
            "image_id": item["image_id"],
            "version": item["version"],
            "image_commit_id": item["image_commit_id"]
        }
        break
    return static_image_info, dynamic_image_info


def _check_result_file_is_need(model_type, file_name, model_list):
    """
    # 判断文件是否需要拷贝
    """
    if model_type == "static_graph":
        model_list = [x.lower() for x in model_list]
        file_name = file_name.lower()
    elif model_type == "dynamic_graph":
        # 去除开头的 dy_
        model_list = [x[3:].lower() for x in model_list]
        file_name = file_name.replace("dynamic_", "").lower()
    for model_name in model_list:
        if file_name.startswith(model_name):
            return True
    return False


def _copy_history_result_to_save_dir(save_dir, image_version, model_type, model_list):
    """
    # 将历史 result 文件直接拷贝至当前持久化目录
    """
    base_dir = os.environ.get("base_dir")
    source_dir = "%s/logs/%s/%s/index/" % (base_dir, image_version, model_type)
    destination_dir = save_dir + "/" + model_type + "/index"
    os.makedirs(destination_dir, exist_ok=True)
    # 遍历 source_dir，根据文件名前缀判断是否符合预期，符合预期则拷贝至 destination_dir
    if not os.path.exists(source_dir):
        return
    for file_name in os.listdir(source_dir):
        need_copy = _check_result_file_is_need(model_type, file_name, model_list)
        if need_copy:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(destination_dir, file_name))


def _check_model_result_file_exists(save_dir, model_type, model_name):
    """
    # 判断 save_dir 中当前判断的模型是否已经存在
    """
    destination_dir = save_dir + "/" + model_type + "/index"
    if not os.path.exists(destination_dir):
        return False
    for file_name in os.listdir(destination_dir):
        if model_type == "static_graph":
            model_name = model_name.lower()
            file_name = file_name.lower()
        elif model_type == "dynamic_graph":
            # 去除开头的 dy_
            model_name = model_name[3:].lower()
            file_name = file_name.replace("dynamic_", "").lower()
        if file_name.startswith(model_name):
            return True
    return False


def _calculate_remain_models(save_dir, output_file, static_models, dynamic_models, static_image_id, dynamic_image_id):
    """
    # 计算有哪些模型仍然还需要重新运行
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = current_dir + "/models.json"
    with open(config_file, "r") as f:
        data = json.loads(f.read())
        all_static_graph = data["static_graph"]
        all_dynamic_graph = data["dynamic_graph"]
    if "all" in static_models:
        static_models = all_static_graph
    if "all" in dynamic_models:
        dynamic_models = all_dynamic_graph
    for model_name in copy.deepcopy(static_models):
        is_exists = _check_model_result_file_exists(save_dir, "static_graph", model_name)
        if is_exists:
            static_models.remove(model_name)
    for model_name in copy.deepcopy(dynamic_models):
        is_exists = _check_model_result_file_exists(save_dir, "dynamic_graph", model_name)
        if is_exists:
            dynamic_models.remove(model_name)
    with open(output_file, "w") as f:
        if static_models:
            f.write("static_graph %s %s\n" % (",".join(static_models), static_image_id))
        if dynamic_models:
            f.write("dynamic_graph %s %s\n" % (",".join(dynamic_models), dynamic_image_id))


def _query_latest_daily_result(args):
    """
    # 查询最近一次daily run的结果
    """
    # 初始化目录
    save_dir = os.path.abspath(args.save_dir)
    output_file = args.output_file
    if args.static_models.strip():
        static_models = [x.strip() for x in args.static_models.strip().split(",")]
    else:
        static_models = []
    if args.dynamic_models.strip():
        dynamic_models = [x.strip() for x in args.dynamic_models.strip().split(",")]
    else:
        dynamic_models = []
    os.makedirs(save_dir, exist_ok=True)
    # query 对应的 image_id
    static_image_info, dynamic_image_info = _query_latest_image_id()
    print("static_image_info: %s" % str(static_image_info))
    print("dynamic_image_info: %s" % str(dynamic_image_info))
    # 动态图和静态图分别拷贝，分别对比
    if static_models:
        _copy_history_result_to_save_dir(save_dir, static_image_info["version"], "static_graph", static_models)
    if dynamic_models:
        _copy_history_result_to_save_dir(save_dir, dynamic_image_info["version"], "dynamic_graph", dynamic_models)
    _calculate_remain_models(
        save_dir, output_file, static_models, dynamic_models,
        static_image_info["version"], dynamic_image_info["version"]
    )


if __name__ == "__main__":
    # main
    args = parser.parse_args()
    _query_latest_daily_result(args)
    # debug
    # print(_query_latest_image_id())
    # /ssd1/ljh/benchmark_ce/70725f72756e/anaconda/bin/python3 query_latest_daily_benchmark_result.py \
    # --save_dir /ssd1/ljh/benchmark_ce/70725f72756e/logs/27/diff/ \
    # --output_file /ssd1/ljh/benchmark_ce/70725f72756e/benchmark/scripts/remain.txt --dynamic_models tsn
