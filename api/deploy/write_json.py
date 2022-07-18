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
import os
import time
import json
import op_benchmark_unit

COMPARE_RESULT_SHOWS = {
    "Better": "Better",
    "Equal": "Equal",
    "Less": "Less",
    "Unknown": "Unknown",
    "Unsupport": "Unsupport",
    "Others": "Others",
    "Total": "Total"
}


def dump_json(benchmark_result_list,
              output_path=None,
              compare_framework=None,
              dump_with_parameters=None):
    """
    dump data json files
    """
    if output_path is None:
        timestamp = time.strftime('%Y-%m-%d', time.localtime(int(time.time())))
        output_path = "op_benchmark_summary-%s-json" % timestamp
        print("Output path is not specified, use %s." % output_path)
    print("-- Write json files into %s." % output_path)

    pwd = os.getcwd()
    save_path = os.path.join(pwd, output_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    json_result = {"paddle": dict(), "pytorch": dict(), "compare": dict()}

    for device in ["cpu", "gpu"]:
        json_result["device"] = device
        for case_id in range(len(benchmark_result_list)):
            op_unit = benchmark_result_list[case_id]
            for direction in ["forward", "backward", "backward_forward"]:
                result = op_unit.get(device, direction)
                time_set = ["total",
                            "gpu_time"] if device == "gpu" else ["total"]
                for key in time_set:
                    compare_result = COMPARE_RESULT_SHOWS.get(
                        result["compare"][key], "--")
                    json_result["compare"][direction + key] = compare_result
                    for framework in ["paddle", compare_framework]:
                        json_result[framework]["case_name"] = op_unit.case_name
                        json_result[framework][direction + key] = result[
                            framework][key]
            with open(save_path + "/" + json_result[framework]["case_name"] +
                      "_" + device + ".json", 'w') as f:
                f.write(json.dumps(json_result))
                f.write("\n")
