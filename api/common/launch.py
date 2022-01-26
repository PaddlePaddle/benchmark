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


class NsightRunnerForDynamicScheduling(object):
    """
    Use Nsight System tool to analyse performance of OP.
    """

    def run(self, cmd, start_step, end_step):
        stdout, exit_code = self._nsight_nvtx(cmd)
        if exit_code == 0:
            parse_status, op_type_list, scheduling_time_dict = self._parse_logs(
                stdout.split("\n"), start_step, end_step)
            if parse_status:
                return scheduling_time_dict
        print("Running Error:\n {}".format(stdout))
        return {}

    def _nsight_nvtx(self, cmd):
        return system.run_command(
            "nsys profile -t cuda,nvtx --stats true -o tmp.qdrep --force-overwrite true {}".
            format(cmd))

    def _to_float(self, s):
        return float(s.replace(',', ''))

    def _calculate_avg_time_per_op(self, l):
        """
        Within a step, the same OP may be executed multiple times. When the information
         within the OP is analyzed, each OP needs to be statistics separately.
        """
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        num_calls = self._to_float(l[2]) - 1
        return (total_time - max_time) / num_calls

    def _calculate_avg_time_per_step(self, l, num_step):
        """
        Within a step, the same OP may be executed multiple times. When the influence
         of this OP to the entire step needs to be analysed, the OP needs to be processed
         as a whole in a step. 
        """
        # The same op may appear multiple times within a step.
        total_time = self._to_float(l[1])
        max_time = self._to_float(l[5])
        return (total_time - max_time) / (num_step - 1)

    def _calculate_scheduling_time(self, outside_time, inside_time):
        if outside_time and inside_time:
            return round(outside_time - inside_time, 2)
        return None

    def _parse_logs(self, logs, start_step, end_step):
        flag_nvtx_time_start = False
        parse_status = False
        nvtx_time_start_step = 0
        total_step_time = 0.0
        total_op_call_time_per_step = 0.0
        # num step of using profile
        num_step = end_step - start_step
        # Profile data in start_step and end_step may be not correct,
        # so we need to select some reliable data. Number of reliable
        # step data is step_count.
        step_count = 0

        op_type_list = []
        # scheduling time:
        # op_type pybind_imperative_func (imperative_avg_time)
        # op_type (fwd_trace_op_avg_time)
        # op_type compute (fwd_op_compute_avg_time)
        # op_type_grad (bwd_trace_op_avg_time)
        # op_type_grad compute (bwd_op_compute_avg_time)
        _nvtx_meta_data_dict = {}
        scheduling_time_dict = {}

        # get the op_type counted in the profile.
        # get the scheduling list that needs to be analyse.
        for i in range(len(logs)):
            line = api_param.parse_string(logs[i])
            if flag_nvtx_time_start:
                infos = line.strip().split()
                if not infos:
                    continue
                nvtx_range_type = infos[-1]
                if nvtx_range_type == 'pybind_imperative_func' or nvtx_range_type == 'compute':
                    op_type = infos[-2]
                    if op_type not in op_type_list and '_grad' not in op_type:
                        op_type_list.append(op_type)
                        _nvtx_meta_data_dict[op_type +
                                             ' pybind_imperative_func'] = None
                        _nvtx_meta_data_dict[op_type] = None
                        _nvtx_meta_data_dict[op_type + ' compute'] = None
                        _nvtx_meta_data_dict[op_type + '_grad'] = None
                        _nvtx_meta_data_dict[op_type + '_grad compute'] = None
            if not flag_nvtx_time_start and 'NVTX Push-Pop Range Statistics:' in line:
                flag_nvtx_time_start = True
                nvtx_time_start_step = i

        # parse report to get meta scheduling time
        for i in range(nvtx_time_start_step, len(logs)):
            line = api_param.parse_string(logs[i])
            infos = line.strip().split()
            if not infos:
                continue
            nvtx_range_type = infos[-1]
            if nvtx_range_type == 'pybind_imperative_func' or nvtx_range_type == 'compute':
                nvtx_range_type = infos[-2] + ' ' + nvtx_range_type

            # step time
            if nvtx_range_type.isdigit() and int(
                    nvtx_range_type) > start_step and int(
                        nvtx_range_type) < end_step - 1:
                step_count += 1
                step_time = self._to_float(infos[1])
                total_step_time += step_time

            if nvtx_range_type in _nvtx_meta_data_dict:
                avg_time = self._calculate_avg_time_per_op(infos)
                _nvtx_meta_data_dict[nvtx_range_type] = round(avg_time, 2)

                if '_grad' in nvtx_range_type and 'compute' not in nvtx_range_type or 'pybind_imperative_func' in nvtx_range_type:
                    total_op_call_time_per_step += self._calculate_avg_time_per_step(
                        infos, num_step)

        # analyse scheduling time
        scheduling_time_dict['step_time'] = round(
            total_step_time / step_count, 2) if step_count != 0 else None
        scheduling_time_dict['op_call_time_per_step'] = round(
            total_op_call_time_per_step, 2)
        scheduling_time_dict[
            'python_call_time'] = self._calculate_scheduling_time(
                scheduling_time_dict['step_time'],
                scheduling_time_dict['op_call_time_per_step'])
        for op_type in op_type_list:
            tmp_op_time_dict = {}
            tmp_op_time_dict['imperative_avg_time'] = _nvtx_meta_data_dict[
                op_type + ' pybind_imperative_func']
            tmp_op_time_dict['fwd_trace_op_avg_time'] = _nvtx_meta_data_dict[
                op_type]
            tmp_op_time_dict['fwd_op_compute_avg_time'] = _nvtx_meta_data_dict[
                op_type + ' compute']
            tmp_op_time_dict['bwd_trace_op_avg_time'] = _nvtx_meta_data_dict[
                op_type + '_grad']
            tmp_op_time_dict['bwd_op_compute_avg_time'] = _nvtx_meta_data_dict[
                op_type + '_grad compute']

            tmp_op_time_dict[
                'imperative_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['imperative_avg_time'],
                    tmp_op_time_dict['fwd_trace_op_avg_time'])
            tmp_op_time_dict[
                'fwd_trace_op_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['fwd_trace_op_avg_time'],
                    tmp_op_time_dict['fwd_op_compute_avg_time'])
            tmp_op_time_dict[
                'bwd_trace_op_call_time'] = self._calculate_scheduling_time(
                    tmp_op_time_dict['bwd_trace_op_avg_time'],
                    tmp_op_time_dict['bwd_op_compute_avg_time'])

            scheduling_time_dict[op_type] = tmp_op_time_dict

        parse_status = True
        return parse_status, op_type_list, scheduling_time_dict


def launch(benchmark_script, benchmark_script_args, with_nvprof=False):
    """
    If with_nvprof is True, it will launch the following command firstly to
    get the gpu_time:
        nvprof python benchmark_script benchmark_script_args

    Then the normal testing command will be launched:
        python benchmark_script benchmark_script_args
    """
    benchmark_args_dict = _args_list_to_dict(benchmark_script_args)
    task = benchmark_args_dict.get("task", "speed")

    if with_nvprof:
        if task == "speed":
            _set_args(benchmark_script_args, "--profiler", "nvprof")
        elif task == "scheduling":
            _set_args(benchmark_script_args, "--profiler", "nvprof_nvtx")
    cmd = "{} {} {}".format(sys.executable, benchmark_script,
                            " ".join(benchmark_script_args))
    if with_nvprof:
        if task == "speed":
            if is_ampere_gpu():
                runner = NsightRunner()
            else:
                runner = NvprofRunner()
            gpu_time = runner.run(cmd)
            _set_args(benchmark_script_args, "--profiler", "none")
            return gpu_time
        elif task == "scheduling":
            framework = benchmark_args_dict.get("framework", "paddle")
            assert framework == "paddle", "framework must be 'paddle' if task is 'scheduling'."
            repeat = int(benchmark_args_dict.get("repeat", "10"))
            # The performance of the first few steps is unstable.
            assert repeat >= 10, "repeat must be greater than 10 if task is scheduling, but received {}.".format(
                repeat)
            runner = NsightRunnerForDynamicScheduling()
            scheduling_time_dict = runner.run(cmd, 5, repeat)
            _set_args(benchmark_script_args, "--profiler", "none")
            return scheduling_time_dict
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


def _set_args(args, arg, value):
    if arg in args:
        for i in range(len(args)):
            if args[i] == arg:
                args[i + 1] = value
                break
    else:
        args.append(arg)
        args.append(value)


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
    only_print = benchmark_args_dict.get("only_print", "False")

    system.check_commit()

    if use_gpu and (task == "speed" or
                    task == "scheduling") and profiler == "none":
        output_time = launch(
            args.benchmark_script,
            args.benchmark_script_args,
            with_nvprof=True)
        if task == "speed":
            args.benchmark_script_args.append(" --gpu_time ")
            args.benchmark_script_args.append(str(output_time))
        if task == "scheduling":
            args.benchmark_script_args.append(" --scheduling_times ")
            args.benchmark_script_args.append("\"" + str(output_time) + "\"")
            _set_args(args.benchmark_script_args, "--only_print", "True")

    launch(
        args.benchmark_script, args.benchmark_script_args, with_nvprof=False)
    if task == "scheduling":
        _set_args(args.benchmark_script_args, "--only_print", str(only_print))
