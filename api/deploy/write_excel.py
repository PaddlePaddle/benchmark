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


def _op_filename(case_name, framework, device, task, direction):
    if direction == "backward_forward":
        direction = "backward"
    filename = case_name + "-" + framework + "_" + device + "_" + task + "_" + direction + ".txt"
    return filename


def _op_result_path(op_result_dir, case_name, framework, device, task,
                    direction):
    filename = _op_filename(case_name, framework, device, task, direction)
    return os.path.abspath(os.path.join(op_result_dir, filename))


def _op_result_url(url_prefix, case_name, framework, device, task, direction):
    filename = _op_filename(case_name, framework, device, task, direction)
    return os.path.join(url_prefix, filename)


def _write_summary_worksheet(benchmark_result_list, worksheet, title_format,
                             cell_formats):
    def _write_summary_unit(compare_result, category, worksheet, row):
        compare_result_keys = compare_result.compare_result_keys
        compare_result_colors = {"Better": "green", "Less": "red"}
        if category is not None:
            worksheet.write(row, 0, category, title_format)
        for col in range(len(compare_result_keys)):
            title = COMPARE_RESULT_SHOWS[compare_result_keys[col]]
            worksheet.write(row, col + 1, title, title_format)

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
                        worksheet.write(row, col + 1, this_str,
                                        cell_formats[color])
                    row += 1
        return row

    column_width = [40, 20, 20, 20, 20, 20, 20]
    for col in range(len(column_width)):
        col_char = chr(ord("A") + col)
        worksheet.set_column(col_char + ":" + col_char, column_width[col])

    row = 0
    # case_level summary
    compare_result_case_level = op_benchmark_unit.summary_compare_result(
        benchmark_result_list)
    row = _write_summary_unit(compare_result_case_level, "case_level",
                              worksheet, row)

    # op_level summary
    compare_result_op_level, compare_result_dict_ops_detail = op_benchmark_unit.summary_compare_result_op_level(
        benchmark_result_list, return_op_detail=True)
    row = _write_summary_unit(compare_result_op_level, "op_level", worksheet,
                              row + 1)

    # summary detail for each op
    for op_type, op_compare_result in sorted(
            compare_result_dict_ops_detail.items()):
        row = _write_summary_unit(op_compare_result, op_type, worksheet,
                                  row + 1)


def _write_speed_accuracy_unit(ws,
                               row,
                               col,
                               case_name,
                               task,
                               content,
                               op_result_dir,
                               framework,
                               device,
                               direction,
                               cell_formats,
                               color,
                               url_prefix=None):
    framework = "paddle" if task == "accuracy" else framework
    op_result_path = _op_result_path(op_result_dir, case_name, framework,
                                     device, task, direction)
    if url_prefix and os.path.exists(op_result_path):
        op_result_url = _op_result_url(url_prefix, case_name, framework,
                                       device, task, direction)
        ws.write_url(
            row,
            col,
            url=op_result_url,
            string=content,
            cell_format=cell_formats[color + "_underline"])
    else:
        ws.write(row, col, content, cell_formats[color])


def _write_compare_result_unit(ws, row, col, compare_result, compare_ratio,
                               cell_formats, color):
    if compare_ratio != "--" and compare_ratio != 0.0:
        if compare_ratio > 2.0:
            compare_result += " (%.2fx)" % (compare_ratio - 1.0)
        elif compare_ratio < 0.5:
            compare_result += " (%.2fx)" % (1.0 / compare_ratio - 1.0)
        else:
            compare_percent = "%.2f" % (abs(1.0 - compare_ratio) * 100)
            compare_result += " (" + compare_percent + "%)"
    ws.write(row, col, compare_result, cell_formats[color])


def _write_detail_worksheet(benchmark_result_list, worksheet, device,
                            direction, op_result_dir, url_prefix,
                            compare_framework, op_frequency_dict, align,
                            title_format, cell_formats):
    def _write_title_and_set_column_width(worksheet, device, compare_framework,
                                          title_format):
        title_names = ["case_name"]
        column_width = [36]
        if op_frequency_dict is not None:
            title_names.append("frequency")
            column_width.append(10)

        time_set = ["total"] if device == "cpu" else ["total", "kernel"]
        for key in time_set:
            title_names.append("paddle(%s)" % key)
            title_names.append(compare_framework + "(%s)" % key)
            title_names.append("compare result")
            column_width.append(16)
            column_width.append(16)
            column_width.append(16)
        title_names.append("accuracy")
        title_names.append("parameters")
        column_width.append(10)
        column_width.append(80)

        for col in range(len(column_width)):
            col_char = chr(ord("A") + col)
            worksheet.set_column(col_char + ":" + col_char, column_width[col])

        for col in range(len(title_names)):
            worksheet.write(0, col, title_names[col], title_format)

    # row 0: titles
    _write_title_and_set_column_width(worksheet, device, compare_framework,
                                      title_format)

    row = 0
    for case_id in range(len(benchmark_result_list)):
        op_unit = benchmark_result_list[case_id]
        if direction in [
                "backward", "backward_forward"
        ] and op_unit.op_type in special_op_list.NO_BACKWARD_OPS:
            continue

        result = op_unit.get(device, direction)

        row += 1
        worksheet.write(row, 0, op_unit.case_name)

        if op_frequency_dict is not None:
            num_frequency = 0
            if op_unit.op_type in op_frequency_dict.keys():
                num_frequency = op_frequency_dict[op_unit.op_type]
            worksheet.write_number(row, 1, num_frequency, align)
            col = 2
        else:
            col = 1

        time_set = ["total"] if device == "cpu" else ["total", "gpu_time"]
        for key in time_set:
            if result["compare"][key] == "Less":
                color = "red"
            elif result["compare"][key] == "Better":
                color = "green"
            else:
                color = "black"

            for framework in ["paddle", compare_framework]:
                op_time = result[framework][key]
                _write_speed_accuracy_unit(
                    worksheet, row, col, op_unit.case_name, "speed", op_time,
                    op_result_dir, framework, device, direction, cell_formats,
                    color, url_prefix)
                col += 1

            compare_result = COMPARE_RESULT_SHOWS.get(result["compare"][key],
                                                      "--")
            compare_ratio = result["compare"][key + "_ratio"]
            _write_compare_result_unit(worksheet, row, col, compare_result,
                                       compare_ratio, cell_formats, color)
            col += 1

        color = "red" if result["accuracy"] in ["False", "false"] else "black"
        difference = result["difference"]
        if difference != "--" and difference != "-" and difference != "0.0":
            difference = "%.2E" % (float(difference))
        _write_speed_accuracy_unit(worksheet, row, col, op_unit.case_name,
                                   "accuracy", difference, op_result_dir,
                                   "paddle", device, direction, cell_formats,
                                   color, url_prefix)

        worksheet.write(row, col + 1, op_unit.parameters)


def dump_excel(benchmark_result_list,
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

    wb = xlw.Workbook(output_path)
    align = wb.add_format({"align": "left"})
    title_format = wb.add_format({
        'bold': True,
        'font_color': 'black',
        'bg_color': '#6495ED'
    })
    cell_formats = {}
    for underline in [False, True]:
        for color in ["green", "red", "black"]:
            key = color + "_underline" if underline else color
            value = wb.add_format({
                'bold': True,
                'underline': underline,
                'font_color': color
            })
            cell_formats[key] = value

    ws = wb.add_worksheet("summary")
    _write_summary_worksheet(benchmark_result_list, ws, title_format,
                             cell_formats)

    if url_prefix:
        print("url prefix: ", url_prefix)
    for device in ["gpu", "cpu"]:
        for direction in ["forward", "backward", "backward_forward"]:
            worksheet_name = device + "_" + direction
            ws = wb.add_worksheet(worksheet_name)
            _write_detail_worksheet(benchmark_result_list, ws, device,
                                    direction, op_result_dir, url_prefix,
                                    compare_framework, op_frequency_dict,
                                    align, title_format, cell_formats)
    wb.close()
