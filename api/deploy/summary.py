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
"""
summary script
"""
from __future__ import print_function

import os
import sys
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import models.benchmark_server.helper as helper
from benchmark_op import models

from common import utils
import op_benchmark_unit

res = {}


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
    file_name = os.path.splitext(os.path.basename(inputfile))[0]
    case_name = file_name.split("-")[0]
    op_type = op_benchmark_unit.parse_op_type(case_name)
    if specified_op_list and op_type not in specified_op_list:
        return

    print("-- Parse %s from %s" % (case_name, inputfile))

    # Add case_name to the global dict.
    if case_name not in res:
        res[case_name] = {}

    statistic_beg_idx = file_name.find("-")
    statistic_type = file_name[statistic_beg_idx + 1:]
    last_line = _read_last_line(inputfile)
    # print(last_line)

    # Parse parameters of current case from the result dict.
    _parse_parameters(case_name, last_line)

    if last_line and "_speed_" in statistic_type:
        _parse_speed(case_name, statistic_type, last_line)

    if last_line and "_accuracy_" in statistic_type:
        _parse_accuracy(case_name, statistic_type, last_line)


def read_frequency_from_text(op_frequency_path):
    op_frequency_dict = {}
    with open(op_frequency_path, "r") as f:
        for line in f.readlines():
            contents = line.split()
            if len(contents) != 3:
                continue
            op_frequency_dict[contents[1]] = int(contents[2])
    return op_frequency_dict


def dump_mysql(data):
    """
    dump data to mysql database
    """
    timestamp = time.time()
    for i in range(len(data)):
        dic = data[i]
        op_record = models.OpRecord2()
        op_record.timestamp = timestamp
        op_record.case_name = dic['name']
        op_record.paddle_cpu_accuracy = dic.get("paddle_cpu_accuracy_forward", "--")
        op_record.paddle_cpu_accuracy_backwards = dic.get("paddle_cpu_accuracy_backward", "--")
        op_record.paddle_gpu_accuracy = dic.get("paddle_gpu_accuracy_forward", "--")
        op_record.paddle_gpu_accuracy_backwards = dic.get("paddle_gpu_accuracy_backward", "--")
        op_record.paddle_cpu_perf = dic.get("paddle_cpu_speed_forward", "--")
        op_record.tf_cpu_perf = dic.get("tensorflow_cpu_speed_forward", "--")
        op_record.paddle_gpu_perf = dic.get("paddle_gpu_speed_forward", "--")
        op_record.tf_gpu_perf = dic.get("tensorflow_gpu_speed_forward", "--")
        op_record.paddle_cpu_perf_backwards = dic.get("paddle_cpu_speed_backward", "--")
        op_record.tf_cpu_perf_backwards = dic.get("tensorflow_cpu_speed_backward", "--")
        op_record.paddle_gpu_perf_backwards = dic.get("paddle_gpu_speed_backward", "--")
        op_record.tf_gpu_perf_backwards = dic.get("tensorflow_gpu_speed_backward", "--")
        op_record.config = dic.get("parameters", "--")
        op_record.gpu_time_backward = dic.get("gpu_time_backward", "--")
        op_record.gpu_time = dic.get("gpu_time", "--")
        op_record.tf_gpu_time_backward = dic.get("tf_gpu_time_backward", "--")
        op_record.tf_gpu_time = dic.get("tf_gpu_time", "--")
        op_record.save()


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
        get_job_res(os.path.join(op_result_dir, filename), specified_op_list)

    data = []
    benchmark_result_list = []
    for key, value in sorted(res.items()):
        case_detail = value.copy()
        case_detail['name'] = key
        data.append(case_detail)

        op_unit = op_benchmark_unit.OpBenchmarkUnit(case_detail)
        # print(op_unit)
        benchmark_result_list.append(op_unit)

    op_frequency_dict = None
    if args.op_frequency_path:
        op_frequency_dict = read_frequency_from_text(args.op_frequency_path)

    if args.dump_to_text:
        import write_text
        write_text.dump_text(benchmark_result_list, args.output_path,
                             args.dump_with_parameters)

    if args.dump_to_excel:
        import write_excel
        write_excel.dump_excel(benchmark_result_list, op_result_dir,
                               args.url_prefix, args.output_path,
                               op_frequency_dict)

    if args.dump_to_mysql:
        try:
            dump_mysql(data)
        except Exception as e:
            print("dump data into mysql failed, please check reason!")
            print(e)
