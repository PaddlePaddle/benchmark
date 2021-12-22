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

import os
import sys
import argparse

from common import system
from common import api_param


def _nvprof(cmd):
    return system.run_command("nvprof --profile-from-start off {}".format(cmd))


def _nsight(cmd):
    return system.run_command(
        "nsys profile -t cublas,cuda,cudnn,nvtx  --capture-range=cudaProfilerApi --stats true {}".
        format(cmd))


def _parse_gpu_time(line):
    infos = line.strip().split()
    percent = float(infos[2].replace("%", "")) * 0.01
    gpu_time = infos[3]
    if gpu_time.endswith("us"):
        gpu_time = float(gpu_time.replace("us", "")) * 0.001
    elif gpu_time.endswith("ms"):
        gpu_time = float(gpu_time.replace("ms", ""))
    elif gpu_time.endswith("s"):
        gpu_time = float(gpu_time.replace("s", "")) * 1000
    else:
        raise ValueError("Invalid time: %s" % gpu_time)
    calls = int(infos[4])
    function = infos[8]
    for i in range(9, len(infos)):
        function = function + " " + infos[i]
    print("percent: %.2f; gpu_time: %.4f ms; calls: %d; function: %s" %
          (percent, gpu_time, calls, function))

    total_gpu_time = gpu_time / percent
    print("total gpu_time: %.4f ms" % total_gpu_time)
    print("")
    return total_gpu_time


def _parse_nvprof_logs(logs):
    line_from = None
    line_to = None
    total_gpu_time = 0.0
    for i in range(len(logs)):
        line = api_param.parse_string(logs[i])
        if "GPU activities:" in line:
            line_from = i - 1
        if line_from is not None and "API calls:" in line:
            line_to = i
    if line_from is not None and line_to is not None:
        for i in range(line_from, line_to):
            print(logs[i])
        print("")
        return True, _parse_gpu_time(logs[line_from + 1])
    else:
        return False, 0.0


def launch(benchmark_script, benchmark_script_args, with_nvprof=False):
    """
    If with_nvprof is True, it will launch the following command firstly to
    get the gpu_time:
        nvprof python benchmark_script benchmark_script_args

    Then the normal testing command will be launched:
        python benchmark_script benchmark_script_args
    """

    def _set_profiler(args, value):
        for i in range(len(args)):
            if args[i] == "--profiler":
                args[i + 1] = value
                break
        if i >= len(args):
            args.append("--profiler")
            args.append(value)

    use_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""
    if with_nvprof and use_gpu:
        _set_profiler(benchmark_script_args, "nvprof")
    cmd = "{} {} {}".format(sys.executable, benchmark_script,
                            " ".join(benchmark_script_args))
    if with_nvprof:
        stdout, exit_code = _nvprof(cmd)
        _set_profiler(benchmark_script_args, "none")
        if exit_code == 0:
            parse_status, gpu_time = _parse_nvprof_logs(stdout.split("\n"))
        else:
            parse_status = False
        if parse_status:
            return gpu_time
        else:
            print("Runing Error:\n {}".format(stdout))
    else:
        stdout, exit_code = system.run_command(cmd)
        print(stdout)
        if exit_code != 0:
            sys.exit(exit_code)
    return 0.0


def _args_list_to_dict(arg_list):
    arg_dict = {}
    for i in range(len(arg_list)):
        if arg_list[i].startswith("--"):
            name = arg_list[i].replace("--", "")
            arg_dict[name] = arg_list[i + 1]
    return arg_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OP Benchmark of PaddlePaddle')

    # positional
    parser.add_argument(
        "benchmark_script",
        type=str,
        help="The full path to operator's benchmark script file. If the task "
        "the speed and GPU is used, nvprof will be used to get the GPU kernel time."
    )

    # rest from the operator benchmark program
    parser.add_argument('benchmark_script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    benchmark_args_dict = _args_list_to_dict(args.benchmark_script_args)
    task = benchmark_args_dict.get("task", "speed")
    use_gpu = system.str2bool(benchmark_args_dict.get("use_gpu", "False"))
    profiler = benchmark_args_dict.get("profiler", "none")
    repeat = benchmark_args_dict.get("repeat", "1")

    system.check_commit()

    if use_gpu and task == "speed" and profiler == "none":
        total_gpu_time = launch(
            args.benchmark_script,
            args.benchmark_script_args,
            with_nvprof=True)
        args.benchmark_script_args.append(" --gpu_time ")
        args.benchmark_script_args.append(str(total_gpu_time))

    launch(
        args.benchmark_script, args.benchmark_script_args, with_nvprof=False)
