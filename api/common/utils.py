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

import sys
import traceback
import numpy as np
import json
import collections
import itertools

from . import special_op_list


class ArrayComparator(object):
    def __init__(self, output, target, atol):
        assert output.shape == target.shape, "The output's shape is expected be the same as target, but receieved %s vs %s." % (
            str(output.shape), str(target.shape))

        self.max_absolute_diff = np.float32(-0.0)
        self.offset = -1
        self.max_relative_diff = np.float32(-0.0)
        self.consistent = True

        self._compare(output.flatten(), target.flatten(), atol)

    def __lt__(self, other):
        if isinstance(other, np.float32) or isinstance(other, float):
            return self.max_absolute_diff < other
        else:
            return self.max_absolute_diff < other.max_absolute_diff

    def __gt__(self, other):
        if isinstance(other, np.float32) or isinstance(other, float):
            return self.max_absolute_diff > other
        else:
            return self.max_absolute_diff > other.max_absolute_diff

    def to_string(self):
        return "max_absolute_diff = %.3e, max_relative_diff = %.3e, offset = %d, %s vs %s" % (
            self.max_absolute_diff, self.max_relative_diff, self.offset,
            str(self.output_diff_value), str(self.target_diff_value))

    def _compare(self, output, target, atol):
        output_fp32 = output.astype(np.float32)
        target_fp32 = target.astype(np.float32)

        # maximum absolute difference
        absolute_diff = np.abs(output_fp32 - target_fp32)
        self.max_absolute_diff = np.max(absolute_diff)
        self.offset = np.argmax(absolute_diff)

        # maximum relative difference
        max_target_value = np.max(np.abs(target_fp32))
        if max_target_value != 0:
            self.max_relative_diff = self.max_absolute_diff / max_target_value
        else:
            self.max_relative_diff = 0.0
        self.consistent = np.allclose(output_fp32, target_fp32, atol=atol)

        self.output_diff_value = output[self.offset]
        self.target_diff_value = target[self.offset]


def _check_type(output, target):
    def _is_numpy_dtype(value):
        if type(value
                ) in [np.float32, np.float16, np.int32, np.int64, np.bool]:
            return True
        else:
            return False

    if _is_numpy_dtype(output):
        output = np.array([output])

    if _is_numpy_dtype(target):
        target = np.array([target])

    if not isinstance(output, np.ndarray) or not isinstance(target,
                                                            np.ndarray):
        raise TypeError(
            "Output argument's type should be numpy.ndarray, but recieved: %s and %s."
            % (str(type(output)), str(type(target))))
    return output, target


def _check_shape(name, output, target, i):
    if name in ["reshape", "squeeze", "unsqueeze", "transpose"]:
        assert output.shape == target.shape, "The %d-the output's shape is different, %s vs %s." % (
            i, str(output.shape), str(target.shape))
        return output, target

    if output.shape != target.shape:
        output_squeezed = np.squeeze(output)
        target_squeezed = np.squeeze(target)
        output_shape_permutations = list(
            itertools.permutations(output_squeezed.shape,
                                   len(output_squeezed.shape)))
        if output_squeezed.shape != target_squeezed.shape and target_squeezed.shape not in output_shape_permutations:
            raise RuntimeError(
                "The %d-the output's shape is different, %s vs %s." % (
                    i, str(output.shape), str(target.shape)))
        else:
            print(
                "---- Warning: The %d-th output's shape is compatible (same after squeezed/permuted), %s vs %s."
                % (i, str(output.shape), str(target.shape)))
        return output_squeezed, target_squeezed
    return output, target


def _permute_order(name, output, target):
    if name in ["reshape", "squeeze", "unsqueeze", "transpose"]:
        return []

    numbers = list(range(len(target.shape)))
    all_permutations = list(itertools.permutations(numbers, len(numbers)))
    choosed_permutations = []
    for permutation in all_permutations:
        permuted_target_shape = []
        for pos in permutation:
            permuted_target_shape.append(target.shape[pos])
        if permuted_target_shape == list(output.shape):
            choosed_permutations.append(permutation)
    return choosed_permutations


def check_outputs(output_list,
                  target_list,
                  testing_mode,
                  name,
                  atol=1E-6,
                  use_gpu=True,
                  backward=False,
                  config_params=None):
    try:
        import tensorflow as tf
    except ImportError:
        pass

    if not isinstance(output_list, list) or not isinstance(target_list, list):
        raise TypeError(
            "input argument's type should be list of numpy.ndarray.")

    consistent = True
    max_diff = np.float32(0.0)
    num_outputs = 0

    if name not in special_op_list.NO_FETCHES_OPS:
        if len(output_list) != len(target_list):
            if len(output_list) > 1 and len(target_list) == 1 and isinstance(
                    target_list[0], list):
                target_list = target_list[0]
            if len(output_list) == 1 and len(target_list) > 1 and isinstance(
                    output_list[0], list):
                output_list = output_list[0]
            assert len(output_list) == len(
                target_list
            ), "Expected the number of outputs to be equal, but recieved: %d vs %d." % (
                len(output_list), len(target_list))

        num_outputs = len(output_list)
        for i in range(num_outputs):
            output = output_list[i]
            target = target_list[i]

            if testing_mode == "static":
                if isinstance(
                        target,
                        tf.python.framework.indexed_slices.IndexedSlicesValue):
                    print(
                        "---- Warning: Th %d-th target's type is IndexedSlicesValue and the check is skipped. "
                        "It will be fixed later." % i)
                    continue

            output, target = _check_type(output, target)
            output, target = _check_shape(name, output, target, i)

            if output.dtype != target.dtype:
                print(
                    "---- Warning: The %d-the output's data type is different, %s vs %s."
                    % (i, str(output.dtype), str(target.dtype)))

            diff_comparator_i = None
            if output.shape == target.shape:
                diff_comparator_i = ArrayComparator(output, target, atol)

            if diff_comparator_i is None or diff_comparator_i > atol:
                # Try to compare output with permuted target.
                choosed_permutations = _permute_order(name, output, target)
                permutation = None
                for permutation_tmp in choosed_permutations:
                    target_transposed = np.transpose(target, permutation_tmp)
                    diff_comparator_i_tmp = ArrayComparator(
                        output, target_transposed, atol)
                    if diff_comparator_i is None or diff_comparator_i > diff_comparator_i_tmp:
                        diff_comparator_i = diff_comparator_i_tmp
                        permutation = permutation_tmp
                if permutation is not None:
                    print(
                        "---- Warning: The %d-th output need permute. The permutation is %s, outputs shape are %s vs %s."
                        % (i, str(permutation), str(output.shape),
                           str(target.shape)))

            if diff_comparator_i > 1E-6 or diff_comparator_i.max_relative_diff > 1E-6:
                print(
                    "---- Warning: The %d-th output (shape: %s, data type: %s) has diff. Detail: %s, atol is %.2e."
                    % (i, str(output.shape), str(output.dtype),
                       diff_comparator_i.to_string(), atol))

            max_diff = diff_comparator_i.max_absolute_diff if diff_comparator_i > max_diff else max_diff
            if max_diff > atol:
                if name in special_op_list.RANDOM_OP_LIST:
                    print(
                        "---- Warning: The %d-th output is not consistent, but %s is a random operator and we see it correct."
                        % (i, name))
                elif testing_mode == "static" and name in special_op_list.DIFF_IMPLEMENTATION_TF_OPS:
                    print(
                        "---- Warning: The implementation of %s is different with tensorflow. "
                        "When the value of inputs are same, paddle choose the second value as the output and "
                        "tensorflow choose the first value as the output." %
                        (name))
                else:
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
