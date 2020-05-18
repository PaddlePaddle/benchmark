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
import special_op_list


def compare(output1, output2, atol):
    if not isinstance(output1, np.ndarray) or not isinstance(output2,
                                                             np.ndarray):
        raise TypeError("input argument's type should be numpy.ndarray.")

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


def check_outputs(list1, list2, name, atol=1e-6):
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError(
            "input argument's type should be list of numpy.ndarray.")

    consistent = True
    max_diff = np.float32(0.0)

    assert len(list1) == len(list2)
    num_outputs = len(list1)
    for i in xrange(num_outputs):
        output1 = list1[i]
        output2 = list2[i]

        max_diff_i, offset_i = compare(output1, output2, atol)
        if max_diff_i > atol:
            print("---- The %d-th output (shape: %s, data type: %s) has diff. "
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
    if not isinstance(result, dict):
        raise TypeError("Input result should be a dict.")

    runtimes = result.get("total", None)
    stable = result.get("stable", None)
    diff = result.get("diff", None)

    for i in range(len(runtimes)):
        runtimes[i] *= 1000

    sorted_runtimes = np.sort(runtimes)
    if len(sorted_runtimes) <= 2:
        begin = 0
        end = len(sorted_runtimes)
    elif len(sorted_runtimes) <= 10:
        begin = 1
        end = len(sorted_runtimes) - 1
    elif len(sorted_runtimes) <= 20:
        begin = 5
        end = len(sorted_runtimes) - 5
    else:
        begin = 10
        end = len(sorted_runtimes) - 10
    avg_runtime = np.average(sorted_runtimes[begin:end])

    if log_level == 0:
        seg_0 = 0
        seg_1 = len(runtimes)
    elif log_level == 1 and len(runtimes) > 20:
        seg_0 = 10
        seg_1 = len(runtimes) - 10
    else:
        # print all times
        seg_0 = 0
        seg_1 = 0
    for i in range(len(runtimes)):
        if i < seg_0 or i >= seg_1:
            print("Iter {0}, Runtime: {1}".format("%4d" % i, "%.5f ms" %
                                                  runtimes[i]))

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
    status["speed"]["total"] = avg_runtime
    print(json.dumps(status))
