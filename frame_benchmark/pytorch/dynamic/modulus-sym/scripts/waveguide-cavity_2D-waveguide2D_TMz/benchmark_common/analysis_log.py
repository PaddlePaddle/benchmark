# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from pdb import line_prefix
import re
import traceback

from numpy import mean, var


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis."
    )
    parser.add_argument("--speed_log_file", type=str, help="json file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_name",
        help="training model_name, transformer_base",
    )
    parser.add_argument("--base_batch_size", type=int, help="base_batch size on gpu")
    parser.add_argument("--run_mode", type=str, default="DP", help="DP|MP|PP")
    parser.add_argument("--fp_item", type=str, help="fp_item:fp16|fp32")
    parser.add_argument("--keyword", type=str, help="Keyword to specify analysis data")
    parser.add_argument(
        "--skip_steps", type=int, default=2, help="The number of steps to be skipped"
    )
    parser.add_argument(
        "--device_num", type=str, default="N1C1", help="device_num:N1C1|N1C8|N4C32"
    )
    args = parser.parse_args()
    return args


def _is_number(num):
    pattern = re.compile(r"^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$")
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


class TimeAnalyzer(object):
    def __init__(self, filename, keyword=None):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword

    def get_iteration_cost(self):
        iteration_costs = []
        with open(self.filename, "r") as f_object:
            lines = f_object.read().splitlines()
            for line in lines:
                if self.keyword not in line:
                    continue
                try:
                    result = None

                    # Distill the string from a line.
                    line = line.strip()
                    line_words = line.split()
                    for i in range(len(line_words) - 1):
                        if line_words[i] == self.keyword:
                            result = float(line_words[i + 1])
                            iteration_costs.append(result)
                    # Distil the result from the picked string.

                except Exception as exc:
                    print("line is: {}; failed".format(line_prefix))

        return mean(iteration_costs[2:])


if __name__ == "__main__":
    args = parse_args()
    run_info = dict()
    run_info["model_branch"] = os.getenv("model_branch")
    run_info["model_commit"] = os.getenv("model_commit")
    run_info["model_name"] = args.model_name
    run_info["batch_size"] = args.base_batch_size
    run_info["fp_item"] = args.fp_item
    if (
        re.match(r"DP.-MP.-PP.", args.run_mode)
        or "DP_MoE_C" in args.run_mode
        or "Sharding_MoE_C" in args.run_mode
        or re.match(r"DP._MP.", args.run_mode)
    ):
        run_info["run_mode"] = "Collective"
    else:
        run_info["run_mode"] = args.run_mode
    run_info["convergence_value"] = 0
    run_info["convergence_key"] = ""
    run_info["ips"] = 0
    run_info["device_num"] = args.device_num
    run_info["model_run_time"] = os.getenv("model_run_time")
    run_info["frame_commit"] = ""
    run_info["frame_version"] = os.getenv("frame_version")
    device_num = args.device_num
    print("---device_num:-", device_num)
    if "C" in device_num:
        index_c = device_num.index("C")
        print("---index_c:-", index_c)
        gpu_num = int(device_num[index_c + 1 : len(device_num)])
    if "X" in device_num:
        index_c = device_num.index("X")
        print("---index_c:-", index_c)
        gpu_num = 1
    print("-----gpu_num:", gpu_num)
    if "pwgan" in args.model_name:
        print("------analysis ", args.model_name)
        args.keyword = "avg_ms:"

    try:
        analyzer = TimeAnalyzer(args.filename, args.keyword)
        run_info["ips"] = analyzer.get_iteration_cost()
        run_info["speed_unit"] = "ms/iteration"

    except Exception:
        traceback.print_exc()
    print(
        "{}".format(json.dumps(run_info))
    )  # it's required, for the log file path  insert to the database
    with open(args.speed_log_file, "w") as f:
        f.write(json.dumps(run_info))
