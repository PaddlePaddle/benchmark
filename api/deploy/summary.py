#!/bin/python
#coding=utf-8
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
"""
summary script
"""
from __future__ import print_function

import os, sys
import json
import time
import argparse

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from common import utils
from common import special_op_list

res = {}


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


def parse_op_type(case_name):
    import re
    return re.sub("_[0-9]*$", "", case_name)


class OpBenchmarkUnit(object):
    def __init__(self, case_detail):
        self.case_name = case_detail["name"]
        self.op_type = parse_op_type(self.case_name)

        if case_detail.get("parameters", None):
            parameters = case_detail["parameters"].encode("utf-8")
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

                accuracy = self._get_case_value(case_detail, "paddle", device,
                                                "accuracy", direction)
                result["accuracy"] = str(accuracy)

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
                total = "Unkown"
        if device == "gpu":
            gpu_time = result["compare"]["gpu_time"]
            if gpu_time == "--":
                if direction == "backward" and self.op_type in special_op_list.NO_BACKWARD_OPS:
                    gpu_time = "Unsupport"
                else:
                    gpu_time = "Unkown"
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
                key = "paddle_" + device + "_accuracy_" + direction
                return case_detail[key]
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


def _read_last_line(inputfile):
    filesize = os.path.getsize(inputfile)
    f = open(inputfile, 'r')

    blocksize = 1024
    if filesize > blocksize:
        maxseekpoint = filesize // blocksize
        f.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        f.seek(0, 0)

    lines = f.readlines()
    if len(lines) >= 2:
        last_line = lines[-2].strip("\n")
    else:
        last_line = None
    f.close()
    return last_line


def _parse_parameters(case_name, last_line):
    assert res.get(case_name, None) is not None

    case_detail = res[case_name]
    if case_detail.get("parameters", None) is None:
        try:
            data = json.loads(last_line)
            param = data['parameters']
            res[case_name]['parameters'] = param.strip("\n")
        except Exception:
            pass


def _parse_speed(case_name, statistic_type, last_line):
    assert res.get(case_name, None) is not None

    gpu_time_key = None
    if statistic_type == "paddle_gpu_speed_forward":
        gpu_time_key = "gpu_time"
    elif statistic_type == "paddle_gpu_speed_backward":
        gpu_time_key = "gpu_time_backward"
    elif statistic_type == "tensorflow_gpu_speed_forward":
        gpu_time_key = "tf_gpu_time"
    elif statistic_type == "tensorflow_gpu_speed_backward":
        gpu_time_key = "tf_gpu_time_backward"

    try:
        data = json.loads(last_line)
        # May set following values:
        #   paddle_cpu/gpu_speed_forward
        #   paddle_cpu/gpu_speed_backward
        #   tensorflow_cpu/gpu_speed_forward
        #   tensorflow_cpu/gpu_speed_backward
        total = data["speed"]["total"]
        total_str = "%.5f" % total
        res[case_name][statistic_type] = total_str
        if gpu_time_key and data["speed"].get("gpu_time", None):
            # May set following values:
            #   gpu_time
            #   gpu_time_backward
            #   tf_gpu_time
            #   tf_gpu_time_backward
            gpu_time = data["speed"]["gpu_time"]
            gpu_time_str = "%.5f" % gpu_time
            res[case_name][gpu_time_key] = gpu_time_str
    except Exception:
        res[case_name][statistic_type] = "--"
        res[case_name][gpu_time_key] = "--"


def _parse_accuracy(case_name, statistic_type, last_line):
    assert res.get(case_name, None) is not None

    try:
        data = json.loads(last_line)
        # May set following values:
        #   paddle_gpu_accuracy_forward
        #   paddle_gpu_accuracy_backward
        #   paddle_cpu_accuracy_forward
        #   paddle_cpu_accuracy_backward
        consitent_status = data['consistent']
        res[case_name][statistic_type] = consitent_status
    except Exception:
        res[case_name][statistic_type] = "--"


def get_job_res(inputfile, specified_op_list=None):
    """
    implements within avoiding too large file

    Content of res:
      name                          - case name, such as 'abs_0'

      paddle_gpu_speed_forward      - GPU runtime, paddle, forward
      paddle_gpu_speed_backward     - GPU runtime, paddle, backward
      tensorflow_gpu_speed_forward  - GPU runtime, tensorflow, forward
      tensorflow_gpu_speed_backward - GPU runtime, tensorflow, backward
      gpu_time                      - GPU kernel time, paddle, forward
      gpu_time_backward             - GPU kernel time, paddle, backward
      tf_gpu_time                   - GPU kernel time, tensorflow, forward
      tf_gpu_time_backward          - GPU kernel time, tensorflow, backward

      paddle_cpu_speed_forward      - CPU runtime, paddle, forward
      paddle_cpu_speed_backward     - CPU runtime, paddle, backward
      tensorflow_cpu_speed_forward  - CPU runtime, tensorflow, forward
      tensorflow_cpu_speed_backward - CPU runtime, tensorflow, backward
   
      paddle_gpu_accuracy_forward   - GPU accuracy status, forward
      paddle_gpu_accuracy_backward  - GPU accuracy status, backward
      paddle_cpu_accuracy_forward   - CPU accuracy status, forward
      paddle_cpu_accuracy_backward  - CPU accuracy status, backward

    Args:
      inputfile (str) -- directory path
    """
    filename = os.path.splitext(os.path.basename(inputfile))[0]
    case_name = filename.split("-")[0]
    op_type = parse_op_type(case_name)
    if specified_op_list and op_type not in specified_op_list:
        return res

    print("-- Parse %s from %s" % (case_name, inputfile))

    # Add case_name to the global dict.
    if case_name not in res:
        res[case_name] = {}

    statistic_beg_idx = filename.find("-")
    statistic_type = filename[statistic_beg_idx + 1:]
    last_line = _read_last_line(inputfile)
    # print(last_line)

    # Parse parameters of current case from the result dict.
    _parse_parameters(case_name, last_line)

    if last_line and "_speed_" in statistic_type:
        _parse_speed(case_name, statistic_type, last_line)

    if last_line and "_accuracy_" in statistic_type:
        _parse_accuracy(case_name, statistic_type, last_line)

    return res


def read_frequency_from_text(op_frequency_path):
    op_frequency_dict = {}
    with open(op_frequency_path, "r") as f:
        for line in f.readlines():
            contents = line.split()
            if len(contents) != 3:
                continue
            op_frequency_dict[contents[1]] = int(contents[2])
    return op_frequency_dict


def summary_compare_result(benchmark_result_list, level="case"):
    compare_result_list = {}
    compare_result_key_list = [
        "Better", "Equal", "Less", "Unkown", "Unsupport", "Total"
    ]
    for result_type in [
            "gpu_forward_total", "gpu_forward_kernel", "gpu_backward_total",
            "gpu_backward_kernel", "cpu_forward_total", "cpu_backward_total"
    ]:
        compare_result_list[result_type] = {}
        for compare_result_key in compare_result_key_list:
            compare_result_list[result_type][compare_result_key] = 0

    for op_unit in benchmark_result_list:
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                result_type = device + "_" + direction

                compare_result_total, compare_result_kernel = op_unit.get_compare_value(
                    device, direction)
                compare_result_list[result_type + "_total"][
                    compare_result_total] += 1
                compare_result_list[result_type + "_total"]["Total"] += 1
                if device == "gpu":
                    compare_result_list[result_type + "_kernel"][
                        compare_result_kernel] += 1
                    compare_result_list[result_type + "_kernel"]["Total"] += 1

    compare_result_str = "%s" % (" ".ljust(24))
    for compare_result_key in compare_result_key_list:
        compare_result_str += "%s" % compare_result_key.ljust(16)
    compare_result_str += "\n"
    for key, value in sorted(compare_result_list.items(), reverse=True):
        compare_result_str += "%s" % key.ljust(24)
        num_total_cases = value["Total"]
        for compare_result_key in compare_result_key_list:
            ratio = float(value[compare_result_key]) / float(num_total_cases)
            ratio_str = "%.2f" % (ratio * 100)
            this_str = "{} ({}%)".format(value[compare_result_key], ratio_str)
            compare_result_str += "%s" % str(this_str).ljust(16)
        compare_result_str += "\n"
    print(compare_result_str)
    return compare_result_list, compare_result_str


def dump_text(benchmark_result_list, output_path, dump_with_parameters):
    if output_path is None:
        timestamp = time.strftime('%Y-%m-%d', time.localtime(int(time.time())))
        output_path = "op_benchmark_summary-%s.txt" % timestamp
        print("Output path is not specified, use %s." % output_path)

    title_total = "%s%s%s" % ("Paddle(total)".ljust(20),
                              "Tensorflow(total)".ljust(20),
                              "status".ljust(10))
    title_kernel = "%s%s%s" % ("Paddle(kernel)".ljust(20),
                               "Tensorflow(kernel)".ljust(20),
                               "status".ljust(10))
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

    _, compare_result_str = summary_compare_result(benchmark_result_list)

    with open(output_path, 'w') as f:
        f.writelines(compare_result_str + "\n")
        for direction in ["Forward", "Backward"]:
            f.writelines(
                "================================================================== %s Running on GPU ==================================================================\n"
                % direction)
            f.writelines(gpu_title.encode("utf-8"))
            f.writelines(output_str_list["gpu_" + direction.lower()].encode(
                "utf-8"))
            f.writelines("\n")

        for direction in ["Forward", "Backward"]:
            f.writelines(
                "========================================== %s Running on CPU =======================================\n"
                % direction)
            f.writelines(cpu_title.encode("utf-8"))
            f.writelines(output_str_list["cpu_" + direction.lower()].encode(
                "utf-8"))
            f.writelines("\n")


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


def dump_excel(benchmark_result_list,
               op_result_dir,
               url_prefix=None,
               output_path=None,
               op_frequency_dict=None):
    """
    dump data to a excel
    """
    import xlsxwriter as xlw

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

    compare_result_list, _ = summary_compare_result(benchmark_result_list)

    summary_ws = wb.add_worksheet("summary")
    summary_column_width = [40, 20, 20, 20, 20, 20, 20]
    for col in range(len(summary_column_width)):
        col_char = chr(ord("A") + col)
        summary_ws.set_column(col_char + ":" + col_char,
                              summary_column_width[col])
    summary_title_names = [
        "Better", "Equal", "Less", "Unkown", "Unsupport", "Total"
    ]
    for col in range(len(summary_title_names)):
        summary_ws.write(0, col + 1, summary_title_names[col], title_format)

    row = 1
    for key, value in sorted(compare_result_list.items(), reverse=True):
        summary_ws.write(row, 0, key)

        col = 1
        num_total_cases = value["Total"]
        for compare_result_key in summary_title_names:
            ratio = float(value[compare_result_key]) / float(num_total_cases)
            ratio_str = "%.2f" % (ratio * 100)
            this_str = "{} ({}%)".format(value[compare_result_key], ratio_str)
            summary_ws.write(row, col, this_str)
            col += 1
        row += 1

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


def dump_mysql(data):
    """
    dump data to mysql database
    """
    timestamp = time.time()
    for i in range(len(data)):
        dic = data[i]
        case_name = dic['name']
        paddle_cpu_accuracy = "--"
        paddle_cpu_accuracy_backwards = "--"
        paddle_gpu_accuracy = "--"
        paddle_gpu_accuracy_backwards = "--"
        paddle_cpu_perf = "--"
        tf_cpu_perf = "--"
        paddle_gpu_perf = "--"
        tf_gpu_perf = "--"
        paddle_cpu_perf_backwards = "--"
        tf_cpu_perf_backwards = "--"
        paddle_gpu_perf_backwards = "--"
        tf_gpu_perf_backwards = "--"
        parameters = "--"
        gpu_time = "--"
        gpu_time_backward = "--"
        tf_gpu_time = "--"
        tf_gpu_time_backward = "--"

        for k, v in dic.items():
            if k == "paddle_cpu_accuracy_forward":
                paddle_cpu_accuracy = v
            elif k == "paddle_cpu_accuracy_backward":
                paddle_cpu_accuracy_backwards = v
            elif k == "paddle_gpu_accuracy_forward":
                paddle_gpu_accuracy = v
            elif k == "paddle_gpu_accuracy_backward":
                paddle_gpu_accuracy_backwards = v
            elif k == "paddle_cpu_speed_forward":
                paddle_cpu_perf = v
            elif k == "tensorflow_cpu_speed_forward":
                tf_cpu_perf = v
            elif k == "paddle_gpu_speed_forward":
                paddle_gpu_perf = v
            elif k == "tensorflow_gpu_speed_forward":
                tf_gpu_perf = v
            elif k == "paddle_cpu_speed_backward":
                paddle_cpu_perf_backwards = v
            elif k == "tensorflow_cpu_speed_backward":
                tf_cpu_perf_backwards = v
            elif k == "paddle_gpu_speed_backward":
                paddle_gpu_perf_backwards = v
            elif k == "tensorflow_gpu_speed_backward":
                tf_gpu_perf_backwards = v
            elif k == "parameters":
                parameters = v
            elif k == "gpu_time_backward":
                gpu_time_backward = v
            elif k == "gpu_time":
                gpu_time = v
            elif k == "tf_gpu_time_backward":
                tf_gpu_time_backward = v
            elif k == "tf_gpu_time":
                tf_gpu_time = v
            else:
                pass

        cmd = 'docker exec mysql ./mysql -e "insert into paddle.op_record2 ' \
              'values(\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\',' \
              '\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', {}, \'{}\', \'{}\', \'{}\', \'{}\');" '\
            .format(
                    case_name, paddle_cpu_accuracy, paddle_cpu_accuracy_backwards, paddle_gpu_accuracy,
                    paddle_gpu_accuracy_backwards, paddle_cpu_perf, tf_cpu_perf, paddle_gpu_perf, tf_gpu_perf,
                    paddle_cpu_perf_backwards, tf_cpu_perf_backwards, paddle_gpu_perf_backwards, tf_gpu_perf_backwards,
                    "--", parameters, timestamp, gpu_time, gpu_time_backward, tf_gpu_time, tf_gpu_time_backward
                    )
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'op_result_dir',
        type=str,
        default=None,
        help='Specify the result directory of operator benchmark.')
    parser.add_argument(
        '--specified_op_list',
        type=str,
        default=None,
        help='Specify the operator list.')
    parser.add_argument(
        '--op_frequency_path',
        type=str,
        default=None,
        help='Specify the path of operator frequency data.')
    parser.add_argument(
        '--dump_to_text',
        type=utils.str2bool,
        default=False,
        help='Whether dumping the summary data to a text file [True|False]')
    parser.add_argument(
        '--dump_to_excel',
        type=utils.str2bool,
        default=False,
        help='Whether dumping summary data to an excel [True|False]')
    parser.add_argument(
        '--dump_with_parameters',
        type=utils.str2bool,
        default=False,
        help='Whether dumping summary data to an text [True|False]')
    parser.add_argument(
        '--url_prefix',
        type=str,
        default=None,
        help='Specify url prefix of output logs.')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Specify the output path.')
    parser.add_argument(
        '--dump_to_mysql',
        type=utils.str2bool,
        default=True,
        help='Whether dumping summary data to mysql database [True|False]')
    args = parser.parse_args()

    op_result_dir = os.path.abspath(args.op_result_dir)
    assert os.path.exists(
        op_result_dir), "Directory %s does not exist." % op_result_dir

    filenames = os.listdir(op_result_dir)
    filenames.remove('api_info.txt')
    assert len(filenames) > 0, "Directory %s is empty." % op_result_dir

    specified_op_list = None
    if args.specified_op_list:
        specified_op_list = args.specified_op_list.split()

    for filename in sorted(filenames):
        res = get_job_res(
            os.path.join(op_result_dir, filename), specified_op_list)

    data = []
    benchmark_result_list = []
    for key, value in sorted(res.items()):
        case_detail = value.copy()
        case_detail['name'] = key
        data.append(case_detail)

        op_unit = OpBenchmarkUnit(case_detail)
        # print(op_unit)
        benchmark_result_list.append(op_unit)

    op_frequency_dict = None
    if args.op_frequency_path:
        op_frequency_dict = read_frequency_from_text(args.op_frequency_path)

    if args.dump_to_text:
        dump_text(benchmark_result_list, args.output_path,
                  args.dump_with_parameters)

    if args.dump_to_excel:
        dump_excel(benchmark_result_list, op_result_dir, args.url_prefix,
                   args.output_path, op_frequency_dict)

    if args.dump_to_mysql:
        try:
            dump_mysql(data)
        except Exception as e:
            print("dump data into mysql failed, please check reason!")
            print(e)
