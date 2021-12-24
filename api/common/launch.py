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


def is_ampere_gpu():
    stdout, exit_code = system.run_command("nvidia-smi -L")
    if exit_code == 0:
        gpu_list = stdout.split("\n")
        if len(gpu_list) >= 1:
            #print(gpu_list[0])
            # GPU 0: NVIDIA A100-SXM4-40GB (UUID: xxxx)
            return gpu_list[0].find("A100") > 0
    return False


class NvprofRunner(object):
    def run(self, cmd):
        stdout, exit_code = self._nvprof(cmd)
        if exit_code == 0:
            parse_status, gpu_time = self._parse_logs(stdout.split("\n"))
            if parse_status:
                return gpu_time
        print("Running Error:\n {}".format(stdout))
        return 0.0

    def _nvprof(self, cmd):
        return system.run_command("nvprof --profile-from-start off {}".format(
            cmd))

    def _parse_logs(self, logs):
        line_from = None
        line_to = None
        for i in range(len(logs)):
            line = api_param.parse_string(logs[i])
            if "GPU activities:" in line:
                line_from = i - 1
            if line_from is not None and "API calls:" in line:
                line_to = i
                break
        if line_from is not None and line_to is not None:
            for i in range(line_from, line_to):
                print(logs[i])
            print("")
            return True, self._parse_gpu_time(logs[line_from + 1])
        else:
            return False, 0.0

    def _parse_gpu_time(self, line):
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
        #print("percent: %.2f; gpu_time: %.4f ms; calls: %d; function: %s" %
        #      (percent, gpu_time, calls, function))

        total_gpu_time = gpu_time / percent
        print("total gpu_time: %.4f ms" % total_gpu_time)
        print("")
        return total_gpu_time


class NsightRunner(object):
    def run(self, cmd):
        stdout, exit_code = self._nsight(cmd)
        if exit_code == 0:
            parse_status, gpu_time = self._parse_logs(stdout.split("\n"))
            if parse_status:
                return gpu_time
        print("Running Error:\n {}".format(stdout))
        return 0.0

    def _nsight(self, cmd):
        return system.run_command(
            "nsys nvprof --profile-from-start=off -o tmp.qdrep {}".format(cmd))

    def _parse_logs(self, logs):
        kernel_line_from = None
        kernel_line_to = None
        memcpy_line_from = None
        memcpy_line_to = None
        for i in range(len(logs)):
            line = api_param.parse_string(logs[i])
            if "CUDA Kernel Statistics:" in line:
                kernel_line_from = i
                for j in range(i + 2, len(logs)):
                    if logs[j] == "":
                        kernel_line_to = j
                        break
            if "CUDA Memory Operation Statistics (by time):" in line:
                memcpy_line_from = i
                for j in range(i + 2, len(logs)):
                    if logs[j] == "":
                        memcpy_line_to = j
                        break

        parse_status = False
        kernel_gpu_time = 0.0
        if kernel_line_from is not None and kernel_line_to is not None:
            for i in range(kernel_line_from, kernel_line_to):
                print(logs[i])
            print("")
            parse_status = True
            kernel_gpu_time = self._parse_gpu_time(logs[kernel_line_from + 4])

        memcpy_gpu_time = 0.0
        if memcpy_line_from is not None and memcpy_line_to is not None:
            for i in range(memcpy_line_from, memcpy_line_to):
                print(logs[i])
            print("")
            parse_status = True
            memcpy_gpu_time = self._parse_gpu_time(logs[memcpy_line_from + 4])

        total_gpu_time = kernel_gpu_time + memcpy_gpu_time
        print(
            "total gpu_time: {:.4f} ms (kernel: {:.4f} ms ({:.2f}%); memcpy: {:.4f} ms ({:.2f}%))".
            format(total_gpu_time, kernel_gpu_time, kernel_gpu_time * 100 /
                   total_gpu_time, memcpy_gpu_time, memcpy_gpu_time * 100 /
                   total_gpu_time))
        print("")
        return parse_status, total_gpu_time

    def _parse_gpu_time(self, line):
        infos = line.strip().split()
        percent = float(infos[0].replace("%", "")) * 0.01
        gpu_time = float(infos[1].replace(",", "")) * 1E-6
        calls = int(infos[2].replace(",", ""))
        function = infos[7]
        for i in range(8, len(infos)):
            function = function + " " + infos[i]
        #print("percent: %.2f; gpu_time: %.4f ms; calls: %d; function: %s" %
        #      (percent, gpu_time, calls, function))
        return gpu_time / percent


def launch(benchmark_script, benchmark_script_args, with_nvprof=False):
    """
    If with_nvprof is True, it will launch the following command firstly to
    get the gpu_time:
        nvprof python benchmark_script benchmark_script_args

    Then the normal testing command will be launched:
        python benchmark_script benchmark_script_args
    """

    def _set_profiler(args, value):
        if "--profiler" in args:
            for i in range(len(args)):
                if args[i] == "--profiler":
                    args[i + 1] = value
                    break
        else:
            args.append("--profiler")
            args.append(value)

    if with_nvprof:
        _set_profiler(benchmark_script_args, "nvprof")
    cmd = "{} {} {}".format(sys.executable, benchmark_script,
                            " ".join(benchmark_script_args))
    if with_nvprof:
        if is_ampere_gpu():
            runner = NsightRunner()
        else:
            runner = NvprofRunner()
        gpu_time = runner.run(cmd)
        _set_profiler(benchmark_script_args, "none")
        return gpu_time
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
    use_gpu = system.str2bool(benchmark_args_dict.get(
        "use_gpu", "False")) and os.environ.get("CUDA_VISIBLE_DEVICES",
                                                None) != ""
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
