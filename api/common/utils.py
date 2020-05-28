#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import traceback
import numpy as np
import json
import collections
import subprocess
import special_op_list


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def run_command(command, shell=True):
    print("run command: %s" % command)
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)

    exit_code = None
    stdout = ''
    while exit_code is None or line:
        exit_code = p.poll()
        line = p.stdout.readline().decode('utf-8')
        stdout += line

    return stdout, exit_code


def _compare(output1, output2, atol):
    max_diff = np.float32(-0.0)
    offset = -1
    try:
        assert len(output1) == len(output2)
        if output1.dtype == np.bool:
            diff = np.array_equal(output1, output2)
            max_diff = np.float32(np.logical_not(diff))
            if diff == False:
                for i in range(len(output1)):
                    if output1[i] != output2[i]:
                        offset = i
            assert np.array_equal(output1, output2)
        else:
            diff = np.abs(output1 - output2)
            max_diff = np.max(diff)
            offset = np.argmax(diff)
            assert np.allclose(output1, output2, atol=atol)
    except (AssertionError) as e:
        pass
    return max_diff, offset


def _check_type(output1, output2):
    def _is_numpy_dtype(value):
        if type(value
                ) in [np.float32, np.float16, np.int32, np.int64, np.bool]:
            return True
        else:
            return False

    if _is_numpy_dtype(output1):
        output1 = np.array([output1])

    if _is_numpy_dtype(output2):
        output2 = np.array([output2])

    if not isinstance(output1, np.ndarray) or not isinstance(output2,
                                                             np.ndarray):
        raise TypeError(
            "Output argument's type should be numpy.ndarray, but recieved: %s and %s."
            % (str(type(output1)), str(type(output2))))
    return output1, output2


def _check_shape(output1, output2):
    shape1 = list(output1.shape)
    shape2 = list(output2.shape)
    if shape1 != shape2:
        print("---- The %d-the output's shape is different, %s vs %s." %
              (i, str(shape1), str(shape2)))
        if shape1 == shape2 + [1]:
            output2 = np.reshape(output2, output1.shape)
        elif shape1 + [1] == shape2:
            output1 = np.reshape(output1, output2.shape)
    return output1, output2


def check_outputs(list1, list2, name, atol=1e-6):
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError(
            "input argument's type should be list of numpy.ndarray.")

    consistent = True
    max_diff = np.float32(0.0)
    num_outputs = 0

    if name not in special_op_list.NO_FETCHES_OPS:
        if len(list1) != len(list2):
            if len(list1) > 1 and len(list2) == 1 and isinstance(list2[0],
                                                                 list):
                list2 = list2[0]
            if len(list1) == 1 and len(list2) > 1 and isinstance(list1[0],
                                                                 list):
                list1 = list1[0]
            assert len(list1) == len(
                list2
            ), "Expected the number of outputs to be equal, but recieved: %d vs %d." % (
                len(list1), len(list2))

        num_outputs = len(list1)
        for i in xrange(num_outputs):
            output1 = list1[i]
            output2 = list2[i]

            output1, output2 = _check_type(output1, output2)
            output1, output2 = _check_shape(output1, output2)

            if output1.dtype != output2.dtype:
                print(
                    "---- The %d-the output's data type is different, %s vs %s."
                    % (i, str(output1.dtype), str(output2.dtype)))

            max_diff_i, offset_i = _compare(output1, output2, atol)
            if max_diff_i > atol:
                print(
                    "---- The %d-th output (shape: %s, data type: %s) has diff. "
                    "The maximum diff is %e, offset is %d: %s vs %s." %
                    (i, str(output1.shape), str(output1.dtype), max_diff_i,
                     offset_i, str(output1.flatten()[offset_i]),
                     str(output2.flatten()[offset_i])))

            max_diff = max_diff_i if max_diff_i > max_diff else max_diff
            if max_diff > atol:
                consistent = False

    status = collections.OrderedDict()
    status["name"] = name
    status["consistent"] = consistent
    status["num_outputs"] = num_outputs
    status["diff"] = max_diff.astype("float")

    if not consistent:
        if name is not None and name in special_op_list.RANDOM_OP_LIST:
            print(
                "---- The output is not consistent, but %s is in the white list."
                % name)
            print(json.dumps(status))
        else:
            print(json.dumps(status))
            assert consistent == True, "The output is not consistent."
    else:
        print(json.dumps(status))


def print_benchmark_result(result, log_level=0):
    assert isinstance(result, dict), "Input result should be a dict."

    runtimes = result.get("total", None)
    walltimes = result.get("wall_time", None)
    gpu_time = result.get("gpu_time", None)
    stable = result.get("stable", None)
    diff = result.get("diff", None)

    repeat = len(runtimes)
    for i in range(repeat):
        runtimes[i] *= 1000
        if walltimes is not None:
            walltimes[i] *= 1000

    sorted_runtimes = np.sort(runtimes)
    if repeat <= 2:
        num_excepts = 0
    elif repeat <= 10:
        num_excepts = 1
    elif repeat <= 20:
        num_excepts = 5
    else:
        num_excepts = 10
    begin = num_excepts
    end = repeat - num_excepts
    avg_runtime = np.average(sorted_runtimes[begin:end])
    if walltimes is not None:
        avg_walltime = np.average(np.sort(walltimes)[begin:end])
    else:
        avg_walltime = 0

    # print all times
    seg_range = [0, 0]
    if log_level == 0:
        seg_range = [0, repeat]
    elif log_level == 1 and repeat > 20:
        seg_range = [10, repeat - 10]
    for i in range(len(runtimes)):
        if i < seg_range[0] or i >= seg_range[1]:
            walltime = walltimes[i] if walltimes is not None else 0
            print("Iter %4d, Runtime: %.5f ms, Walltime: %.5f ms" %
                  (i, runtimes[i], walltime))

    status = collections.OrderedDict()
    status["framework"] = result["framework"]
    status["version"] = result["version"]
    status["name"] = result["name"]
    status["device"] = result["device"]
    if stable is not None and diff is not None:
        status["precision"] = collections.OrderedDict()
        status["precision"]["stable"] = stable
        status["precision"]["diff"] = diff
    status["speed"] = collections.OrderedDict()
    status["speed"]["repeat"] = len(sorted_runtimes)
    status["speed"]["begin"] = begin
    status["speed"]["end"] = end
    status["speed"]["total"] = avg_runtime - avg_walltime
    status["speed"]["wall_time"] = avg_walltime
    status["speed"]["total_include_wall_time"] = avg_runtime
    if gpu_time is not None:
        status["speed"]["gpu_time"] = gpu_time
    print(json.dumps(status))
