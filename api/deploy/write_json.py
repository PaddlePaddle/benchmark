#!/bin/python
# -*- coding: UTF-8 -*-
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

import sys
import six
import json
import op_benchmark_unit

if not six.PY3:
    reload(sys)
    sys.setdefaultencoding("utf8")

COMPARE_RESULT_SHOWS = {
    "Better": "优于",
    "Equal": "打平",
    "Less": "差于",
    "Unknown": "未知",
    "Unsupport": "不支持",
    "Others": "其他",
    "Total": "汇总"
}


def create_summary_json(compare_result, category):
    compare_result_colors = {"Better": "green", "Less": "red"}

    compare_result_keys = compare_result.compare_result_keys
    titles = [category]
    titles.extend(map(COMPARE_RESULT_SHOWS.get, compare_result_keys))

    colors = [None]
    colors.extend(
        map(lambda k: compare_result_colors.get(k, "black"),
            compare_result_keys))

    datas = list()
    for device in ["gpu", "cpu"]:
        for direction in ["forward", "backward"]:
            for method in ["total", "kernel"]:
                if device == "cpu": continue
                data = [
                    "{} {} ({})".format(device.upper(),
                                        direction.capitalize(), method)
                ]
                value = compare_result.get(device, direction, method)
                num_total_cases = value["Total"]
                for compare_result_key in compare_result_keys:
                    num_cases = value[compare_result_key]
                    if num_cases > 0:
                        ratio = float(num_cases) / float(num_total_cases)
                        this_str = "{} ({:.2f}%)".format(num_cases,
                                                         ratio * 100)
                    else:
                        this_str = "--"
                    data.append(this_str)
                datas.append(data)

    return {"title": titles, "color": colors, "data": datas}


def dump_json(benchmark_result_list, output_path=None):
    """
    dump data to a json file
    """
    if output_path is None:
        print("Output path is not specified, will not dump json.")
        return

    compare_result_case_level = op_benchmark_unit.summary_compare_result(
        benchmark_result_list)
    compare_result_op_level = op_benchmark_unit.summary_compare_result_op_level(
        benchmark_result_list)

    with open(output_path, 'w') as f:
        summary_case_json = create_summary_json(compare_result_case_level,
                                                "case_level")
        f.write(json.dumps(summary_case_json) + '\n')
        summary_op_json = create_summary_json(compare_result_op_level,
                                              "case_level")
        f.write(json.dumps(summary_op_json))
