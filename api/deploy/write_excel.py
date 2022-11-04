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

import os
import sys
import time
import string
import xlsxwriter as xlw
import op_benchmark_unit
from common import special_op_list

COMPARE_RESULT_SHOWS = {
    "Better": "优于",
    "Equal": "打平",
    "Less": "差于",
    "Unknown": "未知",
    "Unsupport": "不支持",
    "Others": "其他",
    "Total": "汇总"
}


def _op_filename(case_name, framework, device, task, direction, suffix=""):
    if direction == "backward_forward":
        direction = "backward"
    filename = case_name + "-" + framework + "_" + device + "_" + task + "_" + direction + suffix + ".txt"
    return filename


def _op_result_path(op_result_dir,
                    case_name,
                    framework,
                    device,
                    task,
                    direction,
                    precision="fp32"):
    suffix_list = []
    if precision == "fp32":
        suffix_list = ["", "_fp32"]
    elif precision == "fp16":
        suffix_list = ["_fp16"]

    for suffix in suffix_list:
        filename = _op_filename(case_name, framework, device, task, direction,
                                suffix)
        filepath = os.path.abspath(os.path.join(op_result_dir, filename))
        if os.path.exists(filepath):
            return filepath
    return None


def _get_speed_unit_color(compare_result_str, op_time=None):
    if compare_result_str == "Less":
        color = "red"
    elif compare_result_str == "Better":
        color = "green"
    elif op_time is not None and op_time != "--" and float(op_time) == 0.0:
        color = "blue"
    else:
        color = "black"
    return color


class ExcelWriter(object):
    def __init__(self, output_path, url_prefix, compare_framework):
        self.workbook = xlw.Workbook(output_path)
        self.align = self.workbook.add_format({"align": "left"})
        self.title_format = self.workbook.add_format({
            'bold': True,
            'font_color': 'black',
            'bg_color': '#6495ED'
        })
        self.normal_format = {
            "fp32": self.workbook.add_format({
                "font_color": "black"
            }),
            "fp16": self.workbook.add_format({
                "bg_color": "#F5F5F5"
            })
        }
        self.cell_formats = {}
        for underline in [False, True]:
            for precision in ["fp32", "fp16"]:
                for color in ["green", "red", "black", "blue"]:
                    if underline:
                        key = precision + "_" + color + "_underline"
                    else:
                        key = precision + "_" + color
                    cell_format_dict = {
                        'bold': True,
                        'underline': underline,
                        'font_color': color
                    }
                    if underline and color == "red":
                        cell_format_dict["bg_color"] = "#FFF8DC"
                    elif precision == "fp16":
                        cell_format_dict["bg_color"] = "#F5F5F5"
                    self.cell_formats[key] = self.workbook.add_format(
                        cell_format_dict)
        self.url_prefix = url_prefix
        self.compare_framework = compare_framework

    def close(self):
        self.workbook.close()
        self.workbook = None

    def _write_summary_unit(self, compare_result, precision, category,
                            worksheet, row):
        compare_result_keys = compare_result.compare_result_keys
        compare_result_colors = {"Better": "green", "Less": "red"}
        if category is not None:
            worksheet.write(row, 0, category, self.title_format)
        for col in range(len(compare_result_keys)):
            title = COMPARE_RESULT_SHOWS[compare_result_keys[col]]
            worksheet.write(row, col + 1, title, self.title_format)

        row += 1
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                method_set = ["total"
                              ] if device == "cpu" else ["total", "kernel"]
                for method in method_set:
                    category = device.upper() + " " + string.capwords(
                        direction) + " (" + method + ")"
                    worksheet.write(row, 0, category)

                    value = compare_result.get(device, direction, method)
                    num_total_cases = value["Total"]
                    for col in range(len(compare_result_keys)):
                        compare_result_key = compare_result_keys[col]
                        num_cases = value[compare_result_key]

                        if num_cases > 0:
                            color = compare_result_colors.get(
                                compare_result_key, "black")
                            ratio = float(num_cases) / float(num_total_cases)
                            ratio_str = "%.2f" % (ratio * 100)
                            this_str = "{} ({}%)".format(num_cases, ratio_str)
                        else:
                            color = "black"
                            this_str = "--"
                        worksheet.write(
                            row, col + 1, this_str,
                            self.cell_formats[precision + "_" + color])
                    row += 1
        return row

    def add_summary_worksheet(self, fp32_benchmark_result_list,
                              fp16_benchmark_result_list):
        assert self.workbook is not None

        worksheet = self.workbook.add_worksheet("summary")

        column_width = [30, 20, 20, 20, 20, 20, 20]
        for col in range(len(column_width)):
            col_char = chr(ord("A") + col)
            worksheet.set_column(col_char + ":" + col_char, column_width[col])

        row = -1
        # case_level summary
        for precision in ["fp32", "fp16"]:
            benchmark_result_list = fp32_benchmark_result_list if precision == "fp32" else fp16_benchmark_result_list
            if len(benchmark_result_list) == 0:
                continue
            compare_result_case_level = op_benchmark_unit.summary_compare_result(
                benchmark_result_list)
            row = self._write_summary_unit(
                compare_result_case_level, precision,
                "case_level ({})".format(precision), worksheet, row + 1)

        # op_level summary
        for precision in ["fp32", "fp16"]:
            benchmark_result_list = fp32_benchmark_result_list if precision == "fp32" else fp16_benchmark_result_list
            if len(benchmark_result_list) == 0:
                continue
            compare_result_op_level, compare_result_dict_ops_detail = op_benchmark_unit.summary_compare_result_op_level(
                benchmark_result_list, return_op_detail=True)
            row = self._write_summary_unit(compare_result_op_level, precision,
                                           "op_level ({})".format(precision),
                                           worksheet, row + 1)

        # summary detail for each op
        for op_type, op_compare_result in sorted(
                compare_result_dict_ops_detail.items()):
            for precision in ["fp32", "fp16"]:
                benchmark_result_list = fp32_benchmark_result_list if precision == "fp32" else fp16_benchmark_result_list
                if len(benchmark_result_list) == 0:
                    continue
                row = self._write_summary_unit(
                    op_compare_result, precision,
                    "{} ({})".format(op_type, precision), worksheet, row + 1)

    def _write_speed_accuracy_unit(self, worksheet, precision, row, col,
                                   case_name, task, content, op_result_dir,
                                   framework, device, direction, color):
        framework = "paddle" if task == "accuracy" else framework
        op_result_path = _op_result_path(op_result_dir, case_name, framework,
                                         device, task, direction, precision)
        if self.url_prefix and op_result_path is not None:
            op_result_url = self.url_prefix + "/" + os.path.basename(
                op_result_path)
            worksheet.write_url(
                row,
                col,
                url=op_result_url,
                string=content,
                cell_format=self.cell_formats[precision + "_" + color +
                                              "_underline"])
        else:
            worksheet.write(row, col, content,
                            self.cell_formats[precision + "_" + color])

    def _write_compare_result_unit(self, worksheet, precision, row, col,
                                   compare_result, compare_ratio, color):
        if compare_ratio != "--" and compare_ratio != 0.0:
            # compare_ratio >= 1.0
            compare_ratio = compare_ratio if compare_ratio >= 1.0 else 1.0 / compare_ratio
            if compare_ratio >= 2.0:
                compare_result += " (%.2fx)" % (compare_ratio - 1.0)
            else:  #  1.0 <= compare_ratio < 2.0
                compare_percent = "%.2f" % ((compare_ratio - 1.0) * 100)
                compare_result += " (" + compare_percent + "%)"
        worksheet.write(row, col, compare_result,
                        self.cell_formats[precision + "_" + color])

    def _write_op_frequency(self, worksheet, row, col, op_type,
                            op_frequency_dict):
        num_frequency = 0
        if op_type in op_frequency_dict.keys():
            num_frequency = op_frequency_dict[op_type]
        worksheet.write_number(row, col, num_frequency, self.align)

    def _write_title_and_set_column_width(self, worksheet, device, direction,
                                          op_frequency_dict):
        title_names = ["case_name"]
        column_width = [36]
        if op_frequency_dict is not None:
            title_names.append("frequency")
            column_width.append(10)

        title_names.append("precision")
        column_width.append(10)

        time_set = ["total"] if device == "cpu" else ["total", "kernel"]
        for key in time_set:
            title_names.append("paddle(%s)" % key)
            title_names.append(self.compare_framework + "(%s)" % key)
            title_names.append("compare result")
            column_width.append(16)
            column_width.append(16)
            column_width.append(16)
        if device == "gpu" and direction in ["forward", "backward"]:
            title_names.append("paddle(gflops)")
            title_names.append("paddle(gbs)")
        title_names.append("accuracy")
        title_names.append("parameters")
        if device == "gpu" and direction in ["forward", "backward"]:
            column_width.append(16)
            column_width.append(16)
        column_width.append(10)
        column_width.append(80)

        for col in range(len(column_width)):
            col_char = chr(ord("A") + col)
            worksheet.set_column(col_char + ":" + col_char, column_width[col])

        for col in range(len(title_names)):
            worksheet.write(0, col, title_names[col], self.title_format)

    def add_detail_worksheet(self, fp32_benchmark_result_list,
                             fp16_benchmark_result_list, worksheet_name,
                             device, direction, op_result_dir,
                             op_frequency_dict):
        worksheet = self.workbook.add_worksheet(worksheet_name)

        # row 0: titles
        self._write_title_and_set_column_width(worksheet, device, direction,
                                               op_frequency_dict)

        num_results = max(
            len(fp32_benchmark_result_list), len(fp16_benchmark_result_list))
        if len(fp32_benchmark_result_list) > 0 and len(
                fp16_benchmark_result_list) > 0:
            assert (len(fp32_benchmark_result_list) ==
                    len(fp16_benchmark_result_list))

        precision_list = []
        if len(fp32_benchmark_result_list) > 0:
            precision_list.append("fp32")
        if len(fp16_benchmark_result_list) > 0:
            precision_list.append("fp16")

        row = 0
        for case_id in range(num_results):
            for precision in precision_list:
                if precision == "fp32":
                    op_unit = fp32_benchmark_result_list[case_id]
                elif precision == "fp16":
                    op_unit = fp16_benchmark_result_list[case_id]

                if direction in [
                        "backward", "backward_forward"
                ] and op_unit.op_type in special_op_list.NO_BACKWARD_OPS:
                    continue

                result = op_unit.get(device, direction)

                row += 1
                worksheet.write(row, 0, op_unit.case_name,
                                self.normal_format[precision])

                col = 1
                if op_frequency_dict is not None:
                    self._write_op_frequency(worksheet, row, col,
                                             op_unit.op_type,
                                             op_frequency_dict)
                    col += 1

                worksheet.write(row, col, precision,
                                self.normal_format[precision])
                col += 1

                time_set = ["total"
                            ] if device == "cpu" else ["total", "gpu_time"]
                for key in time_set:
                    for framework in ["paddle", self.compare_framework]:
                        op_time = result[framework][key]
                        color = _get_speed_unit_color(
                            result["compare"][key], op_time=op_time)
                        self._write_speed_accuracy_unit(
                            worksheet, precision, row, col, op_unit.case_name,
                            "speed", op_time, op_result_dir, framework, device,
                            direction, color)
                        col += 1

                    compare_result = COMPARE_RESULT_SHOWS.get(
                        result["compare"][key], "--")
                    compare_ratio = result["compare"][key + "_ratio"]
                    color = _get_speed_unit_color(result["compare"][key])
                    self._write_compare_result_unit(worksheet, precision, row,
                                                    col, compare_result,
                                                    compare_ratio, color)
                    col += 1

                # Write gflops and gbs of paddle, only for gpu_forward and gpu_backward now.
                if device == "gpu" and direction in ["forward", "backward"]:
                    color = _get_speed_unit_color(result["compare"][
                        "gpu_time"])
                    for key in ["gflops", "gbs"]:
                        perf = result["paddle"][key]
                        self._write_speed_accuracy_unit(
                            worksheet, precision, row, col, op_unit.case_name,
                            "speed", perf, op_result_dir, "paddle", device,
                            direction, color)
                        col += 1

                color = "red" if result["accuracy"] in ["False", "false"
                                                        ] else "black"
                difference = result["difference"]
                if difference != "--" and difference != "-" and difference != "0.0":
                    difference = "%.2E" % (float(difference))
                self._write_speed_accuracy_unit(
                    worksheet, precision, row, col, op_unit.case_name,
                    "accuracy", difference, op_result_dir, "paddle", device,
                    direction, color)

                worksheet.write(row, col + 1, op_unit.parameters,
                                self.normal_format[precision])


def dump_excel(fp32_benchmark_result_list,
               fp16_benchmark_result_list,
               op_result_dir,
               url_prefix=None,
               output_path=None,
               compare_framework=None,
               op_frequency_dict=None):
    """
    dump data to a excel
    """
    if output_path is None:
        timestamp = time.strftime('%Y-%m-%d', time.localtime(int(time.time())))
        output_path = "op_benchmark_summary-%s.xlsx" % timestamp
        print("Output path is not specified, use %s." % output_path)
    print("-- Write to %s." % output_path)

    writer = ExcelWriter(output_path, url_prefix, compare_framework)
    writer.add_summary_worksheet(fp32_benchmark_result_list,
                                 fp16_benchmark_result_list)

    num_fp32_gpu_results, num_fp32_cpu_results = op_benchmark_unit.count_results_for_devices(
        fp32_benchmark_result_list)
    num_fp16_gpu_results, num_fp16_cpu_results = op_benchmark_unit.count_results_for_devices(
        fp16_benchmark_result_list)
    print("-- num_fp32_gpu_results={}, num_fp32_cpu_results={}".format(
        num_fp32_gpu_results, num_fp32_cpu_results))
    print("-- num_fp16_gpu_results={}, num_fp16_cpu_results={}".format(
        num_fp16_gpu_results, num_fp16_cpu_results))

    device_types = []
    num_gpu_results = num_fp32_gpu_results + num_fp16_gpu_results
    num_cpu_results = num_fp32_cpu_results + num_fp16_cpu_results
    if num_gpu_results > 0:
        device_types.append("gpu")
    if num_cpu_results > 0:
        device_types.append("cpu")

    if url_prefix:
        print("url prefix: ", url_prefix)
    for device in device_types:
        for direction in ["forward", "backward", "backward_forward"]:
            worksheet_name = device + "_" + direction
            writer.add_detail_worksheet(fp32_benchmark_result_list,
                                        fp16_benchmark_result_list,
                                        worksheet_name, device, direction,
                                        op_result_dir, op_frequency_dict)
    writer.close()
