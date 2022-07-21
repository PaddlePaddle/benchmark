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
import string

COMPARE_RESULT_SHOWS = {
    "Better": "优于",
    "Equal": "打平",
    "Less": "差于",
    "Unknown": "未知",
    "Unsupport": "不支持",
    "Others": "其他",
    "Total": "汇总"
}


def _write_summary(compare_result, category, f):
    compare_result_keys = compare_result.compare_result_keys
    f.write(category.ljust(25))

    for col in range(len(compare_result_keys)):
        title = COMPARE_RESULT_SHOWS[compare_result_keys[col]]
        f.write(title.ljust(25))
    f.write("\n")

    for device in ["gpu", "cpu"]:
        for direction in ["forward", "backward"]:
            method_set = ["total"] if device == "cpu" else ["total", "kernel"]
            for method in method_set:
                category = device.upper() + " " + string.capwords(
                    direction) + " (" + method + ")"
                f.write(category.ljust(25))
                value = compare_result.get(device, direction, method)
                num_total_cases = value["Total"]
                for col in range(len(compare_result_keys)):
                    compare_result_key = compare_result_keys[col]
                    num_cases = value[compare_result_key]
                    if num_cases > 0:
                        ratio = float(num_cases) / float(num_total_cases)
                        ratio_str = "%.2f" % (ratio * 100)
                        this_str = "{} ({}%)".format(num_cases, ratio_str)
                    else:
                        this_str = "--"
                    f.write(this_str.ljust(25))
                f.write("\n")


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
                               compare_framework + "(kernel)".ljust(20),
                               "status".ljust(10))
    title_else = "%s%s" % ("accuracy".ljust(10), "paramaters")
    gpu_title = "%s%s%s%s%s\n" % ("case_id".ljust(12), "case_name".ljust(40),
                                  title_total, title_kernel, title_else)
    cpu_title = "%s%s%s%s\n" % ("case_id".ljust(12), "case_name".ljust(40),
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
        # case_level summary
        compare_result_case_level = op_benchmark_unit.summary_compare_result(
            benchmark_result_list)
        f.writelines("=" * 75 + " Case Level Summary " + "=" * 84 + "\n")
        _write_summary(compare_result_case_level, "case_level", f)

        # op_level summary
        compare_result_op_level, compare_result_dict_ops_detail = op_benchmark_unit.summary_compare_result_op_level(
            benchmark_result_list, return_op_detail=True)
        f.writelines("=" * 75 + " OP Level Summary " + "=" * 86 + "\n")
        _write_summary(compare_result_op_level, "op_level", f)

        # summary detail for each op
        f.writelines("=" * 75 + " Summary Details for Each OP " + "=" * 75 +
                     "\n")
        for op_type, op_compare_result in sorted(
                compare_result_dict_ops_detail.items()):
            _write_summary(op_compare_result, op_type, f)

        # write each op details
        for direction in ["Forward", "Backward"]:
            f.writelines("=" * 75 + " %s Running on GPU " % direction + "=" *
                         80 + "\n")
            f.writelines(gpu_title)
            f.writelines(output_str_list["gpu_" + direction.lower()])
            f.writelines("\n")

        for direction in ["Forward", "Backward"]:
            f.writelines("=" * 75 + " %s Running on CPU " % direction + "=" *
                         80 + "\n")
            f.writelines(cpu_title)
            f.writelines(output_str_list["cpu_" + direction.lower()])
            f.writelines("\n")
