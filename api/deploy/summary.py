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

res = {}


def dump_excel(data):
    """
    dump data to a excel
    """
    import xlsxwriter as xlw

    wb = xlw.Workbook('Operators.xlsx')
    ws = wb.add_worksheet('OP')

    align = wb.add_format({'align': 'right'})
    bold = wb.add_format({'bold': 15, 'color': 'black'})
    wrong_format = wb.add_format({'bold': 8, 'color': 'red', 'align': 'right'})

    ws.set_row(0, 15, bold)
    ws.set_column(0, 1, 15)
    ws.set_column(1, 2, 28)
    ws.set_column(2, 3, 28)
    ws.set_column(3, 4, 22)
    ws.set_column(4, 5, 22)
    ws.set_column(5, 6, 22)
    ws.set_column(6, 7, 22)
    ws.set_column(7, 8, 22)
    ws.set_column(8, 9, 22)
    ws.set_column(9, 10, 22)
    ws.set_column(10, 11, 22)

    ws.set_row(0, 15, align)
    ws.set_column(0, 1, 15, align)
    ws.set_column(1, 2, 28, align)
    ws.set_column(2, 3, 28, align)
    ws.set_column(3, 4, 22, align)
    ws.set_column(4, 5, 22, align)
    ws.set_column(5, 6, 22, align)
    ws.set_column(6, 7, 22, align)
    ws.set_column(7, 8, 22, align)
    ws.set_column(8, 9, 22, align)
    ws.set_column(9, 10, 22, align)
    ws.set_column(10, 11, 22, align)

    row = 0
    column = 0
    ws.write(row, column, 'name')
    ws.write(row, column + 1, 'paddle_cpu_accuracy')
    ws.write(row, column + 2, 'paddle_gpu_accuracy')
    ws.write(row, column + 3, 'paddle_cpu_perf(ms)')
    ws.write(row, column + 4, 'tf_cpu_perf(ms)')
    ws.write(row, column + 5, 'paddle_gpu_perf(ms)')
    ws.write(row, column + 6, 'tf_gpu_perf(ms)')
    ws.write(row, column + 7, 'paddle_cpu_perf_backwards(ms)')
    ws.write(row, column + 8, 'tf_cpu_perf_backwards(ms)')
    ws.write(row, column + 9, 'paddle_gpu_perf_backwards(ms)')
    ws.write(row, column + 10, 'tf_gpu_perf_backwards(ms)')

    row = 1
    column = 0
    for i in range(len(data)):
        for key, value in data[i].items():
            if key == 'name':
                val = value.split("-")[0]
                ws.write(row, column, val)
            if key == 'paddle_cpu_accuracy':
                if not value:
                    ws.write(row, column + 1, value, wrong_format)
                else:
                    ws.write(row, column + 1, value)
            if key == 'paddle_gpu_accuracy':
                if not value:
                    ws.write(row, column + 2, value, wrong_format)
                else:
                    ws.write(row, column + 2, value)
            if key == 'paddle_cpu_perf':
                ws.write_string(row, column + 3, value)
            if key == 'tf_cpu_perf':
                ws.write_string(row, column + 4, value)
            if key == 'paddle_gpu_perf':
                ws.write_string(row, column + 5, value)
            if key == 'tf_gpu_perf':
                ws.write_string(row, column + 6, value)
            if key == 'paddle_cpu_perf_backwards':
                ws.write_string(row, column + 7, value)
            if key == 'tf_cpu_perf_backwards':
                ws.write_string(row, column + 8, value)
            if key == 'paddle_gpu_perf_backwards':
                ws.write_string(row, column + 9, value)
            if key == 'tf_gpu_perf_backwards':
                ws.write_string(row, column + 10, value)
        row += 1

    wb.close()


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
    op_type = case_name.split("_")[0]
    if specified_op_list and op_type not in specified_op_list:
        return res

    print("-- Parse %s from %s" % (case_name, inputfile))

    # Add case_name to the global dict.
    if case_name not in res:
        res[case_name] = {}

    statistic_beg_idx = filename.find("-")
    statistic_type = filename[statistic_beg_idx + 1:]
    last_line = _read_last_line(inputfile)

    # Parse parameters of current case from the result dict.
    _parse_parameters(case_name, last_line)

    if last_line and "_speed_" in statistic_type:
        _parse_speed(case_name, statistic_type, last_line)

    if last_line and "_accuracy_" in statistic_type:
        _parse_accuracy(case_name, statistic_type, last_line)

    return res


def _get_case(case_detail, framework, device, task, direction):
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


def dump_text(data, output_path, dump_with_parameters):
    if output_path is None:
        output_path = "op_benchmark_summary.txt"
        print("Output path is not specified, use op_benchmark_summary.txt.")

    title_total = "%s%s%s" % ("Paddle(total)".ljust(16),
                              "Tensorflow(total)".ljust(20),
                              "status".ljust(10))
    title_kernel = "%s%s%s" % ("Paddle(kernel)".ljust(16),
                               "Tensorflow(kernel)".ljust(20),
                               "status".ljust(10))
    title_else = "%s%s" % ("accuracy".ljust(10), "paramaters")

    gpu_forward_str = ""
    gpu_backward_str = ""
    cpu_forward_str = ""
    cpu_backward_str = ""
    for case_id in range(len(data)):
        case_detail = data[case_id]
        case_name = case_detail["name"]
        for device in ["gpu", "cpu"]:
            for direction in ["forward", "backward"]:
                paddle_total, paddle_gpu_time = _get_case(
                    case_detail, "paddle", device, "speed", direction)
                tf_total, tf_gpu_time = _get_case(case_detail, "tensorflow",
                                                  device, "speed", direction)
                accuracy = _get_case(case_detail, "paddle", device, "accuracy",
                                     direction)

                case_line_total = "%s%s%s" % (
                    paddle_total.ljust(16), tf_total.ljust(20), _compare(
                        paddle_total, tf_total).ljust(10))
                if device == "gpu":
                    case_line_kernel = "%s%s%s" % (
                        paddle_gpu_time.ljust(16), tf_gpu_time.ljust(20),
                        _compare(paddle_gpu_time, tf_gpu_time).ljust(10))
                else:
                    case_line_kernel = ""
                if dump_with_parameters:
                    if case_detail.get("parameters", None):
                        parameters = case_detail["parameters"].encode("utf-8")
                        parameters = parameters.replace(" ", "")
                        parameters = parameters.replace("\n", " ")
                    else:
                        parameters = "--"
                    case_line_else = "%s%s" % (str(accuracy).ljust(10),
                                               parameters)
                else:
                    case_line_else = "%s" % (str(accuracy).ljust(10))

                case_line = "%s%s%s%s%s" % (
                    str(case_id + 1).ljust(8), case_name.ljust(40),
                    case_line_total, case_line_kernel, case_line_else)
                if device == "cpu" and direction == "forward":
                    cpu_forward_str += case_line + "\n"
                elif device == "cpu" and direction == "backward":
                    cpu_backward_str += case_line + "\n"
                elif device == "gpu" and direction == "forward":
                    gpu_forward_str += case_line + "\n"
                elif device == "gpu" and direction == "backward":
                    gpu_backward_str += case_line + "\n"

    with open(output_path, 'w') as f:
        gpu_title = "%s%s%s%s%s\n" % ("case_id".ljust(8),
                                      "case_name".ljust(40), title_total,
                                      title_kernel, title_else)
        f.writelines(
            "============================================================== Forward Running on GPU ==============================================================\n"
        )
        f.writelines(gpu_title.encode("utf-8"))
        f.writelines(gpu_forward_str.encode("utf-8"))
        f.writelines(
            "\n============================================================== Backward Running on GPU =============================================================\n"
        )
        f.writelines(gpu_title.encode("utf-8"))
        f.writelines(gpu_backward_str.encode("utf-8"))

        cpu_title = "%s%s%s%s\n" % ("case_id".ljust(8), "case_name".ljust(40),
                                    title_total, title_else)
        f.writelines(
            "\n======================================== Forward Running on CPU ======================================\n"
        )
        f.writelines(cpu_title.encode("utf-8"))
        f.writelines(cpu_forward_str.encode("utf-8"))
        f.writelines(
            "\n======================================== Backward Running on CPU =====================================\n"
        )
        f.writelines(cpu_title.encode("utf-8"))
        f.writelines(cpu_backward_str.encode("utf-8"))


def dump_mysql(data):
    """
    dump data to mysql database
    """
    timestamp = time.time()
    for i in range(len(data)):
        dic = data[i]
        case_name = dic['name']
        paddle_cpu_accuracy = "--"
        paddle_gpu_accuracy = "--"
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
            if k == "paddle_cpu_accuracy_backward":
                paddle_cpu_accuracy_backwards = v
            if k == "paddle_gpu_accuracy_forward":
                paddle_gpu_accuracy = v
            if k == "paddle_gpu_accuracy_backward":
                paddle_gpu_accuracy_backwards = v
            if k == "paddle_cpu_speed_forward":
                paddle_cpu_perf = v
            if k == "tensorflow_cpu_speed_forward":
                tf_cpu_perf = v
            if k == "paddle_gpu_speed_forward":
                paddle_gpu_perf = v
            if k == "tensorflow_gpu_speed_forward":
                tf_gpu_perf = v
            if k == "paddle_cpu_speed_backward":
                paddle_cpu_perf_backwards = v
            if k == "tensorflow_cpu_speed_backward":
                tf_cpu_perf_backwards = v
            if k == "paddle_gpu_speed_backward":
                paddle_gpu_perf_backwards = v
            if k == "tensorflow_gpu_speed_backward":
                tf_gpu_perf_backwards = v
            if k == "parameters":
                parameters = v
            if k == "gpu_time_backward":
                gpu_time_backward = v
            if k == "gpu_time":
                gpu_time = v
            if k == "tf_gpu_time_backward":
                tf_gpu_time_backward = v
            if k == "tf_gpu_time":
                tf_gpu_time = v

        cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.op_record2 ' \
              'values(\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', {}, \'{}\', \'{}\', \'{}\', \'{}\')' \
              'on duplicate key update case_name=\'{}\', paddle_cpu_accuracy=\'{}\', paddle_cpu_accuracy_backwards=\'{}\', paddle_gpu_accuracy=\'{}\', paddle_gpu_accuracy_backwards=\'{}\', paddle_cpu_perf=\'{}\',' \
              'tf_cpu_perf=\'{}\', paddle_gpu_perf=\'{}\', tf_gpu_perf=\'{}\', paddle_cpu_perf_backwards=\'{}\', tf_cpu_perf_backwards=\'{}\',' \
              'paddle_gpu_perf_backwards=\'{}\', tf_gpu_perf_backwards=\'{}\', log_url= \'{}\', config=\'{}\', timestamp={}, gpu_time=\'{}\', gpu_time_backward=\'{}\', tf_gpu_time=\'{}\', tf_gpu_time_backward=\'{}\';"'\
            .format(case_name, paddle_cpu_accuracy, paddle_cpu_accuracy_backwards, paddle_gpu_accuracy, paddle_gpu_accuracy_backwards,
                    paddle_cpu_perf, tf_cpu_perf, paddle_gpu_perf, tf_gpu_perf, paddle_cpu_perf_backwards,
                    tf_cpu_perf_backwards, paddle_gpu_perf_backwards, tf_gpu_perf_backwards, "--", parameters, timestamp, gpu_time, gpu_time_backward, tf_gpu_time, tf_gpu_time_backward,
                    case_name, paddle_cpu_accuracy, paddle_cpu_accuracy_backwards, paddle_gpu_accuracy, paddle_gpu_accuracy_backwards,
                    paddle_cpu_perf, tf_cpu_perf, paddle_gpu_perf, tf_gpu_perf, paddle_cpu_perf_backwards,
                    tf_cpu_perf_backwards, paddle_gpu_perf_backwards, tf_gpu_perf_backwards, "--", parameters, timestamp, gpu_time, gpu_time_backward, tf_gpu_time, tf_gpu_time_backward
                    )
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--op_result_dir',
        type=str,
        default="result",
        help='Specify the result directory of operator benchmark.')
    parser.add_argument(
        '--specified_op_list',
        type=str,
        default=None,
        help='Specify the operator list.')
    parser.add_argument(
        '--dump_to_text',
        type=utils.str2bool,
        default=False,
        help='Whether dumping the summary data to a text file [True|False]')
    parser.add_argument(
        '--dump_to_excel',
        type=utils.str2bool,
        default=True,
        help='Whether dumping summary data to an excel [True|False]')
    parser.add_argument(
        '--dump_with_parameters',
        type=utils.str2bool,
        default=False,
        help='Whether dumping summary data to an excel [True|False]')
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
    for key, value in sorted(res.items()):
        value_copy = value.copy()
        value_copy['name'] = key
        data.append(value_copy)

    if args.dump_to_text:
        dump_text(data, args.output_path, args.dump_with_parameters)

    if args.dump_to_excel:
        try:
            dump_excel(data)
        except Exception as e:
            print("write excel failed, please check the reason!")
            print(e)

    if args.dump_to_mysql:
        try:
            dump_mysql(data)
        except Exception as e:
            print("dump data into mysql failed, please check reason!")
            print(e)
