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


def compare(output1, output2):
    if not isinstance(output1, np.ndarray) or not isinstance(output2, np.ndarray):
        raise TypeError("input argument's type should be numpy.ndarray.")
   
    assert len(output1) == len(output2)
    assert np.allclose(output1, output2, rtol=1.e-6, atol=0)
    max_diff = np.amax(np.absolute(output1 - output2))
    return max_diff

def check_outputs(list1, list2, name=None):
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError("input argument's type should be list of numpy.ndarray.")

    consistent = True
    max_diff = 0.0
    try:
        assert len(list1) == len(list2)
        for i in xrange(len(list1)):
            output1 = list1[i]
            output2 = list2[i]
            diff = compare(output1, output2)
            max_diff = diff if diff > max_diff else max_diff
    except (AssertionError) as e:
        traceback.print_exc()
        consistent = False
    if name is not None:
        print("{name: \"%s\", consistent: \"%s\", diff: %.5f}" % (name, str(consistent), max_diff))
    else:
        print("{consistent: \"%s\", diff: %.5f}" % (str(consistent), max_diff))

def get_stat(stats, key):
    if stats.get(key, None) is None:
        value = None
    else:
        value = stats[key]
    return value

def calc_avg_time(times, begin, end):
    if times is not None:
        if not isinstance(times, list):
            raise TypeError("Input times should be a list.")
        sorted_times = np.sort(times)
        avg_time = np.average(sorted_times[begin:end])
    else:
        avg_time = 0.0
    return avg_time

def print_stat(stats, log_level=0):
    if not isinstance(stats, dict):
        raise TypeError("Input stats should be a dict.")

    runtimes = stats["total"]
    feed_times = get_stat(stats, "feed")
    fetch_times = get_stat(stats, "fetch")
    compute_times = get_stat(stats, "compute")
    stable = get_stat(stats, "stable")
    diff = get_stat(stats, "diff")

    for i in xrange(len(runtimes)):
        runtimes[i] *= 1000
        if feed_times is not None:
            feed_times[i] *= 1000
        if fetch_times is not None:
            fetch_times[i] *= 1000
        if compute_times is not None:
            compute_times[i] *= 1000

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

    avg_feed_time = calc_avg_time(feed_times, begin, end)
    avg_fetch_time = calc_avg_time(fetch_times, begin, end)
    avg_compute_time = calc_avg_time(compute_times, begin, end)

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
    for i in xrange(len(runtimes)):
        if i < seg_0 or i >= seg_1:
            print("Iter {0}, Runtime: {1}".format("%4d" % i, "%.5f ms" % runtimes[i]))

    print("{")
    print("  framework: \"%s\"," % stats["framework"])
    print("  version: \"%s\"," % stats["version"])
    print("  name: \"%s\"," % stats["name"])
    print("  device: \"%s\"," % stats["device"])
    if stable is not None and diff is not None:
        print("  precision: { stable: \"%s\", diff: %.5f }," % (str(stable), diff))
    print("  speed: { repeat: %d, start: %d, end: %d, total: %.5f, feed: %.5f, compute: %.5f, fetch: %.5f }"
              % (len(sorted_runtimes), begin, end, avg_runtime, avg_feed_time, avg_compute_time, avg_fetch_time))
    print("}")
