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

from __future__ import print_function

import os, sys

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from common import special_op_list, api_param


def parse_op_type(case_name):
    import re
    return re.sub("_[0-9]*$", "", case_name)


def _compare(time1, time2):
    try:
        ratio = float(time1) / float(time2)
        if float(time1) > 0 and float(time2) < 0:
            result_str = "Less"
        elif ratio <= 0.95:
            result_str = "Better"
        elif ratio >= 1.05:
            result_str = "Less"
        else:
            result_str = "Equal"
        return result_str
    except Exception:
        return "--"


class OpBenchmarkUnit(object):
    def __init__(self, case_detail):
        self.case_name = case_detail["name"]
        self.op_type = parse_op_type(self.case_name)

        if case_detail.get("parameters", None):
            parameters = api_param.parse_string(case_detail["parameters"])
            parameters = parameters.replace(" ", "")
            parameters = parameters.replace("\n", " ")
        else:
            parameters = "--"
        self.parameters = parameters

        self.gpu_forward = {}
        self.gpu_backward = {}
        self.cpu_forward = {}
        self.cpu_backward = {}
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                attr_name = device + "_" + direction
                result = getattr(self, attr_name)

                paddle_total, paddle_gpu_time = self._get_case_value(
                    case_detail, "paddle", device, "speed", direction)
                result["paddle"] = {
                    "total": paddle_total,
                    "gpu_time": paddle_gpu_time
                }

                tf_total, tf_gpu_time = self._get_case_value(
                    case_detail, "tensorflow", device, "speed", direction)
                result["tensorflow"] = {
                    "total": tf_total,
                    "gpu_time": tf_gpu_time
                }
                result["compare"] = {
                    "total": _compare(paddle_total, tf_total),
                    "gpu_time": _compare(paddle_gpu_time, tf_gpu_time)
                }

                accuracy, difference = self._get_case_value(
                    case_detail, "paddle", device, "accuracy", direction)
                result["accuracy"] = str(accuracy)
                result["difference"] = str(difference)

    def __str__(self):
        debug_str = "case_name    : " + self.case_name + "\n"
        debug_str += "op_type      : " + self.op_type + "\n"
        debug_str += "gpu_forward  : " + str(self.gpu_forward) + "\n"
        debug_str += "gpu_backward : " + str(self.gpu_backward) + "\n"
        debug_str += "cpu_forward  : " + str(self.cpu_forward) + "\n"
        debug_str += "cpu_backward : " + str(self.cpu_backward) + "\n"
        return debug_str

    def to_string(self, device, direction, with_parameters):
        attr_name = device + "_" + direction
        result = getattr(self, attr_name)

        case_line = "%s" % self.case_name.ljust(40)
        time_set = ["total"] if device == "cpu" else ["total", "gpu_time"]
        for key in time_set:
            case_line += "%s%s%s" % (result["paddle"][key].ljust(20),
                                     result["tensorflow"][key].ljust(20),
                                     result["compare"][key].ljust(10))
        case_line += "%s" % result["accuracy"].ljust(10)
        if with_parameters:
            case_line += parameters
        return case_line

    def get(self, device, direction):
        attr_name = device + "_" + direction
        return getattr(self, attr_name)

    def get_compare_value(self, device, direction):
        attr_name = device + "_" + direction
        result = getattr(self, attr_name)

        total = result["compare"]["total"]
        if total == "--":
            if direction == "backward" and self.op_type in special_op_list.NO_BACKWARD_OPS:
                total = "Unsupport"
            else:
                total = "Unknown"
        if device == "gpu":
            gpu_time = result["compare"]["gpu_time"]
            if gpu_time == "--":
                if direction == "backward" and self.op_type in special_op_list.NO_BACKWARD_OPS:
                    gpu_time = "Unsupport"
                else:
                    gpu_time = "Unknown"
            return total, gpu_time
        else:
            return total, None

    def _get_case_value(self, case_detail, framework, device, task, direction):
        assert framework in ["paddle", "tensorflow"]
        assert device in ["cpu", "gpu"]
        assert task in ["speed", "accuracy"]
        assert direction in ["forward", "backward"]

        if task == "accuracy":
            try:
                accuracy_key = "paddle_" + device + "_accuracy_" + direction
                difference_key = "paddle_" + device + "_difference_" + direction
                return case_detail[accuracy_key], case_detail[difference_key]
            except Exception:
                return "--"
        else:
            try:
                total_key = framework + "_" + device + "_speed_" + direction
                if device == "cpu":
                    return case_detail[total_key], "--"

                framework_alias = "" if framework == "paddle" else "tf_"
                direction_alias = "" if direction == "forward" else "_backward"
                gpu_time_key = framework_alias + "gpu_time" + direction_alias
                return case_detail[total_key], case_detail[gpu_time_key]
            except Exception:
                return "--", "--"


class CompareResult(object):
    def __init__(self, compare_result_keys=None):
        if compare_result_keys:
            self.compare_result_keys = compare_result_keys
        else:
            self.compare_result_keys = [
                "Better", "Equal", "Less", "Unknown", "Unsupport", "Total"
            ]
        self.gpu_forward_total = self._create_zero_dict()
        self.gpu_forward_kernel = self._create_zero_dict()
        self.gpu_backward_total = self._create_zero_dict()
        self.gpu_backward_kernel = self._create_zero_dict()
        self.cpu_forward_total = self._create_zero_dict()
        self.cpu_backward_total = self._create_zero_dict()

    def get(self, device, direction, method="total"):
        if device == "cpu":
            assert method == "total"
        attr_name = device + "_" + direction + "_" + method
        return getattr(self, attr_name)

    def _create_zero_dict(self):
        zero_dict = {}
        for key in self.compare_result_keys:
            zero_dict[key] = 0
        return zero_dict

    def to_string(self, category=None):
        if category is None:
            category_width = 24
            compare_result_str = "%s" % (" ".ljust(category_width))
        else:
            category_width = 24 if len(category) < 24 else len(category) + 4
            compare_result_str = "%s" % (category.ljust(category_width))

        content_width = 16
        for compare_result_key in compare_result_keys:
            compare_result_str += "%s" % compare_result_key.ljust(
                content_width)
        compare_result_str += "\n"

        for key, value in sorted(compare_result_list.items(), reverse=True):
            compare_result_str += "%s" % key.ljust(category_width)
            num_total_cases = value["Total"]
            for compare_result_key in compare_result_keys:
                ratio = float(value[compare_result_key]) / float(
                    num_total_cases)
                ratio_str = "%.2f" % (ratio * 100)
                this_str = "{} ({}%)".format(value[compare_result_key],
                                             ratio_str)
                compare_result_str += "%s" % str(this_str).ljust(content_width)
            compare_result_str += "\n"
        return compare_result_str


def summary_compare_result(benchmark_result_list, op_type=None):
    compare_result = CompareResult()

    for op_unit in benchmark_result_list:
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                total, kernel = op_unit.get_compare_value(device, direction)
                compare_result.get(device, direction, "total")[total] += 1
                compare_result.get(device, direction, "total")["Total"] += 1
                if device == "gpu":
                    compare_result.get(device, direction,
                                       "kernel")[kernel] += 1
                    compare_result.get(device, direction,
                                       "kernel")["Total"] += 1
    return compare_result


def summary_compare_result_op_level(benchmark_result_list,
                                    return_op_detail=False):
    benchmark_result_dict = {}
    for op_unit in benchmark_result_list:
        op_type = op_unit.op_type
        if op_type not in benchmark_result_dict.keys():
            benchmark_result_dict[op_type] = []
        benchmark_result_dict[op_type].append(op_unit)

    compare_result_keys = [
        "Better", "Less", "Unknown", "Unsupport", "Others", "Total"
    ]
    compare_result_op_level = CompareResult(compare_result_keys)
    op_type_dict = {}
    for key in [
            "gpu_forward_total", "gpu_forward_kernel", "gpu_backward_total",
            "gpu_backward_kernel", "cpu_forward_total", "cpu_backward_total"
    ]:
        op_type_dict[key] = {}
        for result_key in compare_result_keys:
            op_type_dict[key][result_key] = []

    compare_result_dict_detail = {}
    for op_type, result in sorted(benchmark_result_dict.items()):
        compare_result = summary_compare_result(result, op_type)
        compare_result_dict_detail[op_type] = compare_result

        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                method_set = ["total"
                              ] if device == "cpu" else ["total", "kernel"]
                for method in method_set:
                    value = compare_result.get(device, direction, method)
                    target = compare_result_op_level.get(device, direction,
                                                         method)
                    if value["Better"] == value["Total"]:
                        result_key = "Better"
                    elif value["Less"] == value["Total"]:
                        result_key = "Less"
                    elif value["Unknown"] == value["Total"]:
                        result_key = "Unknown"
                    elif value["Unsupport"] == value["Total"]:
                        result_key = "Unsupport"
                    else:
                        result_key = "Others"
                    op_type_dict[device + "_" + direction + "_" + method][
                        result_key].append(op_type)
                    target[result_key] += 1
                    target["Total"] += 1

    for key, value in op_type_dict.items():
        print(key)
        for result_key, op_list in value.items():
            print("    %s (%3d): %s" %
                  (result_key.ljust(10), len(op_list), ",".join(op_list)))
        print("")

    if return_op_detail:
        return compare_result_op_level, compare_result_dict_detail
    else:
        return compare_result_op_level
