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

import six
import os, sys
import traceback
import numpy as np
import json
import collections
import subprocess
import itertools

if six.PY3:
    from . import special_op_list
else:
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


def check_commit():
    try:
        current_dir = os.getcwd()
        print("-- Current directory: %s" % current_dir)

        dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
        print("-- Entering %s" % dir_of_this_file)
        os.chdir(dir_of_this_file)
        print("-- Current directory: %s" % os.getcwd())
        benchmark_commit, _ = run_command("git rev-parse HEAD")
        benchmark_commit = benchmark_commit.replace("\n", "")
        benchmark_update_time, _ = run_command("git show -s --format=%ad")
        benchmark_update_time = benchmark_update_time.replace("\n", "")
        os.chdir(current_dir)
        print("-- Current directory: %s" % os.getcwd())

        import paddle
        paddle_version = paddle.version.full_version
        paddle_commit = paddle.version.commit

        import tensorflow as tf
        tf_version = tf.__version__

        print(
            "==========================================================================="
        )
        print("-- paddle version             : %s" % paddle_version)
        print("-- paddle commit              : %s" % paddle_commit)
        print("-- tensorflow version         : %s" % tf_version)
        print("-- benchmark commit           : %s" % benchmark_commit)
        print("-- benchmark last update time : %s" % benchmark_update_time)
        print(
            "==========================================================================="
        )
    except Exception:
        pass


def _compare(output1, output2, atol):
    max_diff = np.float32(-0.0)
    offset = -1
    assert output1.shape == output2.shape, "Outputs' shape must be the same, but receieved %s vs %s." % (
        str(output1.shape), str(output2.shape))
    try:
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


def _check_shape(name, output1, output2, i):
    if name in ["reshape", "squeeze", "unsqueeze", "transpose"]:
        assert output1.shape == output2.shape, "The %d-the output's shape is different, %s vs %s." % (
            i, str(output1.shape), str(output2.shape))
        return output1, output2

    if output1.shape != output2.shape:
        output1_squeezed = np.squeeze(output1)
        output2_squeezed = np.squeeze(output2)
        shape1_permutations = list(
            itertools.permutations(output1_squeezed.shape,
                                   len(output1_squeezed.shape)))
        if output1_squeezed.shape != output2_squeezed.shape and output2_squeezed.shape not in shape1_permutations:
            raise RuntimeError(
                "The %d-the output's shape is different, %s vs %s." % (
                    i, str(output1.shape), str(output2.shape)))
        else:
            print(
                "---- Warning: The %d-th output's shape is compatible (same after squeezed/permuted), %s vs %s."
                % (i, str(output1.shape), str(output2.shape)))
        return output1_squeezed, output2_squeezed
    return output1, output2


def _permute_order(name, output1, output2):
    if name in ["reshape", "squeeze", "unsqueeze", "transpose"]:
        return []

    numbers = list(range(len(output2.shape)))
    all_permutations = list(itertools.permutations(numbers, len(numbers)))
    choosed_permutations = []
    for permutation in all_permutations:
        permuted_shape2 = []
        for pos in permutation:
            permuted_shape2.append(output2.shape[pos])
        if permuted_shape2 == list(output1.shape):
            choosed_permutations.append(permutation)
    return choosed_permutations


def check_outputs(list1,
                  list2,
                  name,
                  atol=1E-6,
                  use_gpu=True,
                  backward=False,
                  config_params=None):
    import tensorflow as tf
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
        for i in range(num_outputs):
            output1 = list1[i]
            output2 = list2[i]

            if isinstance(
                    output2,
                    tf.python.framework.indexed_slices.IndexedSlicesValue):
                print(
                    "---- Warning: The type of tensorflow output is IndexedSlicesValue,"
                    "Skip all check, It will be fixed later.")
                continue

            output1, output2 = _check_type(output1, output2)
            output1, output2 = _check_shape(name, output1, output2, i)

            if output1.dtype != output2.dtype:
                print(
                    "---- Warning: The %d-the output's data type is different, %s vs %s."
                    % (i, str(output1.dtype), str(output2.dtype)))

            max_diff_i = None
            offset_i = 0
            if output1.shape == output2.shape:
                max_diff_i, offset_i = _compare(output1, output2, atol)
                output1_diff_value = output1.flatten()[offset_i]
                output2_diff_value = output2.flatten()[offset_i]

            if max_diff_i is None or max_diff_i > atol:
                choosed_permutations = _permute_order(name, output1, output2)
                permutation = None
                for permutation_tmp in choosed_permutations:
                    output2_transposed = np.transpose(output2, permutation_tmp)
                    max_diff_i_tmp, offset_i_tmp = _compare(
                        output1, output2_transposed, atol)
                    if max_diff_i is None or max_diff_i > max_diff_i_tmp:
                        max_diff_i = max_diff_i_tmp
                        offset_i = offset_i_tmp
                        permutation = permutation_tmp
                        output1_diff_value = output1.flatten()[offset_i]
                        output2_diff_value = output2_transposed.flatten()[
                            offset_i]
                if permutation is not None:
                    print(
                        "---- Warning: The %d-th output need permute. The permutation is %s, outputs shape are %s vs %s."
                        % (i, str(permutation), str(output1.shape),
                           str(output2.shape)))

            if max_diff_i > 1E-6:
                print(
                    "---- Warning: The %d-th output (shape: %s, data type: %s) has diff. "
                    "The maximum diff is %e, offset is %d: %s vs %s. atol is %.2e."
                    % (i, str(output1.shape), str(output1.dtype), max_diff_i,
                       offset_i, str(output1_diff_value),
                       str(output2_diff_value), atol))

            max_diff = max_diff_i if max_diff_i > max_diff else max_diff
            if max_diff > atol:
                consistent = False

    status = collections.OrderedDict()
    status["name"] = name
    status["device"] = "GPU" if use_gpu else "CPU"
    status["backward"] = backward
    status["consistent"] = consistent
    status["num_outputs"] = num_outputs
    status["diff"] = max_diff.astype("float")
    status["parameters"] = config_params

    if not consistent:
        if name is not None and name in special_op_list.RANDOM_OP_LIST:
            print(
                "---- Warning: The output is not consistent, but %s is in the white list."
                % name)
            print(json.dumps(status))
            return
        else:
            print("Error: The output is not consistent!!!\n")

    # Make sure the json result is the last line.
    print(json.dumps(status))
    if not consistent:
        sys.exit(1)


def print_benchmark_result(result, log_level=0, config_params=None):
    assert isinstance(result, dict), "Input result should be a dict."

    status = collections.OrderedDict()
    status["framework"] = result["framework"]
    status["version"] = result["version"]
    status["name"] = result["name"]
    status["device"] = result["device"]
    status["backward"] = result["backward"]

    runtimes = result.get("total", None)
    if runtimes is None:
        status["parameters"] = config_params
        print(json.dumps(status))
        return

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

    if avg_runtime - avg_walltime > 0.001:
        total = avg_runtime - avg_walltime
    else:
        print(
            "Average runtime (%.5f ms) is less than average walltime (%.5f ms)."
            % (avg_runtime, avg_walltime))
        total = 0.001

    if stable is not None and diff is not None:
        status["precision"] = collections.OrderedDict()
        status["precision"]["stable"] = stable
        status["precision"]["diff"] = diff
    status["speed"] = collections.OrderedDict()
    status["speed"]["repeat"] = repeat
    status["speed"]["begin"] = begin
    status["speed"]["end"] = end
    status["speed"]["total"] = total
    status["speed"]["wall_time"] = avg_walltime
    status["speed"]["total_include_wall_time"] = avg_runtime
    if gpu_time is not None:
        status["speed"]["gpu_time"] = gpu_time / repeat
    status["parameters"] = config_params
    print(json.dumps(status))
