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

import sys
import argparse

sys.path.append("..")
from common import utils


def nvprof(cmd):
    return utils.run_command("nvprof {}".format(cmd))


def parse_gpu_time(line):
    print("nvprof result: %s" % line)
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
    return total_gpu_time


def launch(framework,
           benchmark_script,
           benchmark_script_args,
           with_nvprof=False):
    cmd = "{} {} {}".format(sys.executable, benchmark_script,
                            " ".join(benchmark_script_args))
    if with_nvprof:
        stdout, exit_code = nvprof(cmd)
        if exit_code == 0:
            logs = stdout.split("\n")
            tag = "GPU activities:"
            gpu_time = 0
            for line in logs:
                if tag in line.encode("utf-8"):
                    total_gpu_time = parse_gpu_time(line.encode("utf-8"))
                    break
            return total_gpu_time
        else:
            print("stdout: {}".format(stdout))
    else:
        stdout, exit_code = utils.run_command(cmd)
        print(stdout)
        if exit_code != 0:
            raise RuntimeError("Run command (%s) error." % cmd)
    return 0.0


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
    task = "speed"
    framework = "paddle"
    use_gpu = False
    for i in range(len(args.benchmark_script_args)):
        if args.benchmark_script_args[i] == "--task":
            task = args.benchmark_script_args[i + 1]
        elif args.benchmark_script_args[i] == "--framework":
            framework = args.benchmark_script_args[i + 1]
        elif args.benchmark_script_args[i] == "--use_gpu":
            use_gpu = utils.str2bool(args.benchmark_script_args[i + 1])

    if use_gpu and task == "speed":
        total_gpu_time = launch(
            framework,
            args.benchmark_script,
            args.benchmark_script_args,
            with_nvprof=True)
        args.benchmark_script_args.append(" --gpu_time ")
        args.benchmark_script_args.append(str(total_gpu_time))

    launch(
        framework,
        args.benchmark_script,
        args.benchmark_script_args,
        with_nvprof=False)
