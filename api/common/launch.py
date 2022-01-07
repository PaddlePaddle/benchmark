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

from common import env
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


class TimeUnit(object):
    def __init__(self):
        self.kernel_time = 0.0
        self.memory_time = 0.0
        self.memcpy_h2d = 0.0
        self.memcpy_d2h = 0.0
        self.memcpy_d2d = 0.0
        self.memset = 0.0

    def total(self):
        self.memory_time = self.memcpy_h2d + self.memcpy_d2d + self.memset
        if not env.benchmark_need_fetch():
            # Normally DtoH is fetching results.
            self.memory_time += self.memcpy_d2h
        return self.kernel_time + self.memory_time

    def __str__(self):
        total_time = self.total()
        if env.benchmark_need_fetch():
            infostr = "total gpu_time (exclude DtoH): {:.4f} ms ".format(
                total_time)
        else:
            infostr = "total gpu_time: {:.4f} ms ".format(total_time)
        if total_time > 0.0:
            infostr += "(kernel: {:.4f} ms ({:.2f}%); memory: {:.4f} ms ({:.2f}%))".format(
                self.kernel_time, self.kernel_time * 100 / total_time,
                self.memory_time, self.memory_time * 100 / total_time)
        else:
            infostr += "(kernel: {:.4f} ms; memory: {:.4f} ms)".format(
                self.kernel_time, self.memory_time)
        infostr += "\n"
        return infostr

    def add_info(self, time, name):
        if name == "[CUDA memcpy HtoD]":
            self._update_memory_time("memcpy_h2d", time)
        elif name == "[CUDA memcpy DtoH]":
            self._update_memory_time("memcpy_d2h", time)
        elif name == "[CUDA memcpy DtoD]":
            self._update_memory_time("memcpy_d2d", time)
        elif name == "[CUDA memset]":
            self._update_memory_time("memset", time)
        else:
            self.kernel_time += time

    def _update_memory_time(self, member_name, time):
        assert member_name in [
            "memcpy_h2d", "memcpy_d2h", "memcpy_d2d", "memset"
        ]
        setattr(self, member_name, time)
        if member_name != "memcpy_d2h" or not env.benchmark_need_fetch():
            self.memory_time += time


class NvprofRunner(object):
    def run(self, cmd, profile_from_start=False):
        stdout, exit_code = self._nvprof(cmd, profile_from_start)
        if exit_code == 0:
            parse_status, gpu_time = self._parse_logs(stdout.split("\n"))
            if parse_status:
                return gpu_time
        print("Running Error:\n {}".format(stdout))
        return 0.0

    def _nvprof(self, cmd, profile_from_start):
        if profile_from_start:
            profile_cmd = "nvprof {}".format(cmd)
        else:
            profile_cmd = "nvprof --profile-from-start off {}".format(cmd)
        return system.run_command(profile_cmd)

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
            time_unit = TimeUnit()
            for i in range(line_from, line_to):
                print(logs[i])
                if i >= line_from + 1:
                    begin_pos = 2 if i == line_from + 1 else 0
                    gpu_time, percent, function = self._parse_line(logs[i],
                                                                   begin_pos)
                    time_unit.add_info(gpu_time, function)
            print("")
            print(time_unit)
            return True, time_unit.total()
        else:
            return False, 0.0

    def _to_millisecond(self, timestr):
        if timestr.endswith("us"):
            return float(timestr.replace("us", "")) * 0.001
        elif timestr.endswith("ms"):
            return float(timestr.replace("ms", ""))
        elif timestr.endswith("s"):
            return float(timestr.replace("s", "")) * 1000
        else:
            raise ValueError("Invalid time: %s" % gpu_time)

    def _parse_line(self, line, begin_pos=0):
        infos = line.strip().split()
        percent = float(infos[begin_pos].replace("%", "")) * 0.01
        gpu_time = self._to_millisecond(infos[begin_pos + 1])
        calls = int(infos[begin_pos + 2])
        function = infos[begin_pos + 6]
        for i in range(begin_pos + 7, len(infos)):
            function = function + " " + infos[i]
        return gpu_time, percent, function


class NsightRunner(object):
    def run(self, cmd, profile_from_start=False):
        stdout, exit_code = self._nsight(cmd, profile_from_start)
        if exit_code == 0:
            parse_status, gpu_time = self._parse_logs(stdout.split("\n"))
            if parse_status:
                return gpu_time
        print("Running Error:\n {}".format(stdout))
        return 0.0

    def _nsight(self, cmd, profile_from_start):
        if profile_from_start:
            profile_cmd = "nsys nvprof -o tmp.qdrep {}".format(cmd)
        else:
            profile_cmd = "nsys nvprof --profile-from-start=off -o tmp.qdrep {}".format(
                cmd)
        return system.run_command(profile_cmd)

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
        if total_gpu_time != 0.0:
            print(
                "total gpu_time: {:.4f} ms (kernel: {:.4f} ms ({:.2f}%); memcpy: {:.4f} ms ({:.2f}%))".
                format(total_gpu_time, kernel_gpu_time, kernel_gpu_time * 100 /
                       total_gpu_time, memcpy_gpu_time, memcpy_gpu_time * 100 /
                       total_gpu_time))
        else:
            print(
                "total gpu_time: {:.4f} ms (kernel: {:.4f} ms; memcpy: {:.4f} ms)".
                format(total_gpu_time, kernel_gpu_time, memcpy_gpu_time))
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


def launch(benchmark_script,
           benchmark_script_args,
           with_nvprof=False,
           profile_from_start=True):
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

    if with_nvprof and not profile_from_start:
        _set_profiler(benchmark_script_args, "nvprof")
    cmd = "{} {} {}".format(sys.executable, benchmark_script,
                            " ".join(benchmark_script_args))
    if with_nvprof:
        if is_ampere_gpu():
            runner = NsightRunner()
        else:
            runner = NvprofRunner()
        gpu_time = runner.run(cmd, profile_from_start)
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
    framework = benchmark_args_dict.get("framework", "paddle")
    use_gpu = system.str2bool(benchmark_args_dict.get(
        "use_gpu", "False")) and os.environ.get("CUDA_VISIBLE_DEVICES",
                                                None) != ""
    profiler = benchmark_args_dict.get("profiler", "none")
    repeat = benchmark_args_dict.get("repeat", "1")

    system.check_commit()

    if use_gpu and task == "speed" and profiler == "none":
        profile_from_start = False
        total_gpu_time = launch(
            args.benchmark_script,
            args.benchmark_script_args,
            with_nvprof=True,
            profile_from_start=profile_from_start)
        args.benchmark_script_args.append(" --gpu_time ")
        args.benchmark_script_args.append(str(total_gpu_time))

    launch(
        args.benchmark_script, args.benchmark_script_args, with_nvprof=False)
