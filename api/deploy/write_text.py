#!/bin/python

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

import time
import op_benchmark_unit


def dump_text(benchmark_result_list,
              output_path=None,
              compare_framework=None,
              dump_with_parameters=None):
    """
    dump data to a text
    """
    if output_path is None:
        timestamp = time.strftime('%Y-%m-%d', time.localtime(int(time.time())))
        output_path = "op_benchmark_summary-%s.txt" % timestamp
        print("Output path is not specified, use %s." % output_path)
    print("-- Write to %s." % output_path)

    title_total = "%s%s%s" % ("paddle(total)".ljust(20),
                              (compare_framework + "(total)").ljust(20),
                              "status".ljust(10))
    title_kernel = "%s%s%s" % ("paddle(kernel)".ljust(20),
                               "Pytorch(kernel)".ljust(20), "status".ljust(10))
    title_else = "%s%s" % ("accuracy".ljust(10), "paramaters")
    gpu_title = "%s%s%s%s%s\n" % ("case_id".ljust(8), "case_name".ljust(40),
                                  title_total, title_kernel, title_else)
    cpu_title = "%s%s%s%s\n" % ("case_id".ljust(8), "case_name".ljust(40),
                                title_total, title_else)

    output_str_list = {
        "gpu_forward": "",
        "gpu_backward": "",
        "cpu_forward": "",
        "cpu_backward": ""
    }

    for case_id in range(len(benchmark_result_list)):
        op_unit = benchmark_result_list[case_id]
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                key = device + "_" + direction
                case_line = "%s%s" % (
                    str(case_id + 1).ljust(8),
                    op_unit.to_string(device, direction, dump_with_parameters))
                output_str_list[key] += case_line + "\n"

    with open(output_path, 'w') as f:
        for direction in ["Forward", "Backward"]:
            f.writelines("=" * 72 + " %s Running on GPU " % direction + "=" *
                         72 + "\n")
            f.writelines(gpu_title)
            f.writelines(output_str_list["gpu_" + direction.lower()])
            f.writelines("\n")

        for direction in ["Forward", "Backward"]:
            f.writelines("=" * 72 + " %s Running on CPU " % direction + "=" *
                         72 + "\n")
            f.writelines(cpu_title)
            f.writelines(output_str_list["cpu_" + direction.lower()])
            f.writelines("\n")
