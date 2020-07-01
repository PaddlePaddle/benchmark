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

import os
import xlsxwriter as xlw
import op_benchmark_unit


def _op_filename(case_name, framework, device, task, direction):
    filename = case_name + "-" + framework + "_" + device + "_" + task + "_" + direction + ".txt"
    return filename


def _op_result_path(op_result_dir, case_name, framework, device, task,
                    direction):
    filename = _op_filename(case_name, framework, device, task, direction)
    return os.path.abspath(os.path.join(op_result_dir, filename))


def _op_result_url(url_prefix, case_name, framework, device, task, direction):
    filename = _op_filename(case_name, framework, device, task, direction)
    return os.path.join(url_prefix, filename)


def _write_summary_worksheet(benchmark_result_list, workbook, title_format):
    compare_result_list, _ = op_benchmark_unit.summary_compare_result(
        benchmark_result_list)
    op_benchmark_unit.summary_compare_result_for_op(benchmark_result_list)

    ws = workbook.add_worksheet("summary")
    column_width = [40, 20, 20, 20, 20, 20, 20]
    for col in range(len(column_width)):
        col_char = chr(ord("A") + col)
        ws.set_column(col_char + ":" + col_char, column_width[col])
    title_names = ["Better", "Equal", "Less", "Unkown", "Unsupport", "Total"]
    for col in range(len(title_names)):
        ws.write(0, col + 1, title_names[col], title_format)

    row = 1
    for key, value in sorted(compare_result_list.items(), reverse=True):
        ws.write(row, 0, key)

        col = 1
        num_total_cases = value["Total"]
        for compare_result_key in title_names:
            ratio = float(value[compare_result_key]) / float(num_total_cases)
            ratio_str = "%.2f" % (ratio * 100)
            this_str = "{} ({}%)".format(value[compare_result_key], ratio_str)
            ws.write(row, col, this_str)
            col += 1
        row += 1


def dump_excel(benchmark_result_list,
               op_result_dir,
               url_prefix=None,
               output_path=None,
               op_frequency_dict=None):
    """
    dump data to a excel
    """
    if output_path is None:
        timestamp = time.strftime('%Y-%m-%d', time.localtime(int(time.time())))
        output_path = "op_benchmark_summary-%s.xlsx" % timestamp
        print("Output path is not specified, use %s." % output_path)

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

    _write_summary_worksheet(benchmark_result_list, wb, title_format)

    if url_prefix:
        print("url prefix: ", url_prefix)
    for device in ["gpu", "cpu"]:
        for direction in ["forward", "backward"]:
            worksheet_name = device + "_" + direction
            ws = wb.add_worksheet(worksheet_name)

            title_names = ["case_name"]
            column_width = [40]
            if op_frequency_dict is not None:
                title_names.append("frequency")
                column_width.append(10)

            time_set = ["total"] if device == "cpu" else ["total", "kernel"]
            for key in time_set:
                title_names.append("Paddle(%s)" % key)
                title_names.append("Tensorflow(%s)" % key)
                title_names.append("status")
                column_width.append(20)
                column_width.append(20)
                column_width.append(10)
            title_names.append("accuracy")
            title_names.append("parameters")
            column_width.append(10)
            column_width.append(80)

            for col in range(len(column_width)):
                col_char = chr(ord("A") + col)
                ws.set_column(col_char + ":" + col_char, column_width[col])

            for col in range(len(title_names)):
                ws.write(0, col, title_names[col], title_format)

            for case_id in range(len(benchmark_result_list)):
                op_unit = benchmark_result_list[case_id]
                result = op_unit.get(device, direction)

                row = case_id + 1
                ws.write(row, 0, op_unit.case_name)

                if op_frequency_dict is not None:
                    num_frequency = 0
                    if op_unit.op_type in op_frequency_dict.keys():
                        num_frequency = op_frequency_dict[op_unit.op_type]
                    ws.write_number(row, 1, num_frequency, align)
                    col = 2
                else:
                    col = 1

                time_set = ["total"
                            ] if device == "cpu" else ["total", "gpu_time"]
                for key in time_set:
                    if result["compare"][key] == "Less":
                        color = "red"
                    elif result["compare"][key] == "Better":
                        color = "green"
                    else:
                        color = "black"

                    for framework in ["paddle", "tensorflow"]:
                        op_time = result[framework][key]
                        op_speed_path = _op_result_path(
                            op_result_dir, op_unit.case_name, framework,
                            device, "speed", direction)
                        if url_prefix and os.path.exists(op_speed_path):
                            op_speed_url = _op_result_url(
                                url_prefix, op_unit.case_name, framework,
                                device, "speed", direction)
                            ws.write_url(
                                row,
                                col,
                                url=op_speed_url,
                                string=op_time,
                                cell_format=cell_formats[color + "_underline"])
                        else:
                            ws.write(row, col, op_time, cell_formats[color])
                        col += 1

                    ws.write(row, col, result["compare"][key],
                             cell_formats[color])
                    col += 1

                op_acc_path = _op_result_path(op_result_dir, op_unit.case_name,
                                              "paddle", device, "accuracy",
                                              direction)
                if url_prefix and os.path.exists(op_acc_path):
                    op_acc_url = _op_result_url(url_prefix, op_unit.case_name,
                                                "paddle", device, "accuracy",
                                                direction)
                    ws.write_url(
                        row,
                        col,
                        url=op_acc_url,
                        string=result["accuracy"],
                        cell_format=cell_formats["black_underline"])
                else:
                    ws.write(row, col, result["accuracy"],
                             cell_formats["black"])
                ws.write(row, col + 1, op_unit.parameters)
    wb.close()
