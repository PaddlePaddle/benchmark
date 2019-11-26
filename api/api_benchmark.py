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

import abc
import time
import traceback
import contextlib
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler


@contextlib.contextmanager
def profile_context(name, use_gpu, profile):
    if profile:
        profile_type = "All" if use_gpu else "CPU"
        with profiler.profiler(profile_type, 'total', name + ".profile"):
            yield
    else:
        yield


class APIBenchmarkBase(object):
    __metaclass__ = abc.ABCMeta      

    def __init__(self):
        self.name = self.__class__.__name__
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.scope = fluid.Scope()
        self.place = None
        self.feed_vars = None
        self.fetch_vars = None
        self.feed_tensors = {}

    @abc.abstractmethod
    def build_program(self, backward=False):
        pass

    def run_with_executor(self, use_gpu, feed=None, repeat=1, log_level=0, check_output=False, profile=False):
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(self.place)
        executor.run(self.startup_program)

        if feed is None:
            feed = self._feed_random_data(as_lodtensor=False)

        runtimes = []
        fetches = []
        with profile_context(self.name, use_gpu, profile):
            for i in xrange(repeat):
                begin = time.time()
                output = executor.run(program=self.main_program,
                                      feed=feed,
                                      fetch_list=self.fetch_vars,
                                      use_program_cache=True,
                                      return_numpy=False)
                end = time.time()
                runtimes.append(end - begin)
                if check_output:
                    fetches.append(output)
        if check_output:
            stable, max_diff = self._check_consistency(fetches)
            stats = { "total": runtimes, "stable": stable, "diff": max_diff }
        else:
            stats = { "total": runtimes }
        stats["device"] = "GPU" if use_gpu else "CPU"
        self._print_stat(stats, log_level=log_level)

    def run_with_core_executor(self, use_gpu, feed=None, repeat=1, log_level=0, check_output=False, profile=False):
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(self.place)
        executor.run(self.startup_program)

        # Use to run main_program
        place = fluid.core.Place()
        place.set_place(self.place)
        core_executor = fluid.core.Executor(place)

        fetch_list_str = []
        for var in self.fetch_vars:
            fetch_list_str.append(var.name)
        ctx = core_executor.prepare(
                    self.main_program.desc, 0, fetch_list_str, False)
        core_executor.create_variables(self.main_program.desc, self.scope, 0)
 
        if feed is None:
            feed = self._feed_random_data(as_lodtensor=False)

        feed_times = []
        fetch_times = []
        compute_times = []
        runtimes = []
        fetches = []
        with profile_context(self.name, use_gpu, profile):
            for i in xrange(repeat):
                begin = time.time()
                self._init_feed_tensor(feed)
                feed_end = time.time()
                core_executor.run_prepared_ctx(ctx, self.scope, False, False, False)
                compute_end = time.time()
                output = self._get_fetch_tensor()
                fetch_end = time.time()

                runtimes.append(fetch_end - begin)
                feed_times.append(feed_end - begin)
                compute_times.append(compute_end - feed_end)
                fetch_times.append(fetch_end - compute_end)
                
                if check_output:
                    fetches.append(output)
        if check_output:
            stable, max_diff = self._check_consistency(fetches)
            stats = {"total": runtimes,
                     "feed": feed_times,
                     "compute": compute_times,
                     "fetch": fetch_times,
                     "stable": stable,
                     "diff": max_diff }
        else:
            stats = { "total": runtimes, "feed": feed_times, "compute": compute_times, "fetch": fetch_times }
        stats["device"] = "GPU" if use_gpu else "CPU"
        self._print_stat(stats, log_level=log_level)

    def _print_stat(self, stats, log_level=0):
        def _get_stat(stats, key):
            if stats.get(key, None) is None:
                value = None
            else:
                value = stats[key]
            return value

        def _calc_avg_time(times, begin, end):
            if times is not None:
                if not isinstance(times, list):
                    raise TypeError("Input times should be a list.")
                sorted_times = np.sort(times)
                avg_time = np.average(sorted_times[begin:end])
            else:
                avg_time = 0.0
            return avg_time

        if not isinstance(stats, dict):
            raise TypeError("Input stats should be a dict.")

        runtimes = stats["total"]
        feed_times = _get_stat(stats, "feed")
        fetch_times = _get_stat(stats, "fetch")
        compute_times = _get_stat(stats, "compute")
        stable = _get_stat(stats, "stable")
        diff = _get_stat(stats, "diff")

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

        avg_feed_time = _calc_avg_time(feed_times, begin, end)
        avg_fetch_time = _calc_avg_time(fetch_times, begin, end)
        avg_compute_time = _calc_avg_time(compute_times, begin, end)

        if log_level == 1:
            for i in xrange(len(runtimes)):
                print("Iter {0}, Runtime: {1}".format("%4d" % i, "%.5f ms" % runtimes[i]))

        print("{")
        print("  name: \"%s\"," % self.name)
        print("  device: \"%s\"," % stats["device"])
        if stable is not None and diff is not None:
            print("  precision: { stable: \"%s\", diff: %.5f }," % (str(stable), diff))
        print("  speed: { repeat: %d, start: %d, end: %d, total: %.5f, feed: %.5f, compute: %.5f, fetch: %.5f }"
                  % (len(sorted_runtimes), begin, end, avg_runtime, avg_feed_time, avg_compute_time, avg_fetch_time))
        print("}")

    def _convert_dtype(self, dtype, to_string=True):
        def _trans(to_string, dtype_str, np_dtype):
            dtype = dtype_str if to_string else np.dtype(np_dtype)
            return dtype

        if not isinstance(dtype, fluid.core.VarDesc.VarType):
            raise TypeError("dtype is not of type fluid.core.VarDesc.VarType")
        if dtype == fluid.core.VarDesc.VarType.FP32:
            return _trans(to_string, "float32", np.float32)
        elif dtype == fluid.core.VarDesc.VarType.FP64:
            return _trans(to_string, "float64", np.float64)
        elif dtype == fluid.core.VarDesc.VarType.FP16:
            return _trans(to_string, "float16", np.float16)
        elif dtype == fluid.core.VarDesc.VarType.INT32:
            return _trans(to_string, "int32", np.int32)
        elif dtype == fluid.core.VarDesc.VarType.INT16:
            return _trans(to_string, "int16", np.int16)
        elif dtype == fluid.core.VarDesc.VarType.INT64:
            return _trans(to_string, "int64", np.int64)
        elif dtype == fluid.core.VarDesc.VarType.BOOL:
            return _trans(to_string, "bool", np.bool)
        elif dtype == fluid.core.VarDesc.VarType.INT16:
            return _trans(to_string, "uint16", np.uint16)
        elif dtype == fluid.core.VarDesc.VarType.UINT8:
            return _trans(to_string, "uint8", np.uint8)
        elif dtype == fluid.core.VarDesc.VarType.INT8:
            return _trans(to_string, "int8", np.int8)
        else:
            raise ValueError("Unsupported dtype %s" % dtype)

    def _feed_random_data(self, as_lodtensor=False):
        print("feed random data")
        feed = {}
        place = fluid.CPUPlace()
        #place = fluid.CUDAPinnedPlace()
        #place = self.place
        for var in self.feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")
                
            shape = var.shape
            dtype = self._convert_dtype(var.dtype, to_string=True)
            data = np.random.random(shape).astype(dtype)
            if as_lodtensor:
                tensor = fluid.core.LoDTensor()
                tensor.set(data, place)
                feed[var.name] = tensor
            else:
                feed[var.name] = data
        return feed

    def _check_outputs(self, output1, output2):
        if not isinstance(output1, np.ndarray) or not isinstance(output2, np.ndarray):
            raise TypeError("output's type should be numpy.ndarray.")
       
        assert len(output1) == len(output2)
        assert np.allclose(output1, output2, rtol=1.e-6, atol=0)
        max_diff = np.amax(np.absolute(output1 - output2))
        return max_diff

    def _check_consistency(self, fetches):
        def _self_check(output):
            if isinstance(output, fluid.core.LoDTensor):
                if output._is_initialized():
                    output = np.array(output)
                else:
                    raise RuntimeError("output tensor is not initialized.")

            if not isinstance(output, np.ndarray):
                raise TypeError("output's type should be numpy.ndarray.")

            if (np.isnan(output)).any():
                raise ValueError("NAN in output.")

            if (np.isinf(output)).any():
                raise ValueError("INF in output.")

            return output

        if not isinstance(fetches, list):
            raise TypeError("fetches is not a list.")

        if len(fetches) <= 0:
            raise ValueError("The number of fetched results is {} (<= 0).".format(len(fetches)))

        stable = True
        repeat = len(fetches)
        num_outputs = len(fetches[0])
        max_diff = 0.0
        for j in xrange(num_outputs):
            if not stable:
                break
            output_0 = None
            for i in xrange(repeat):
                try:
                    output_i = _self_check(fetches[i][j])
                    if i == 0:
                        output_0 = output_i
                    diff = self._check_outputs(output_0, output_i)
                    max_diff = diff if diff > max_diff else max_diff
                except (RuntimeError, ValueError, AssertionError) as e:
                    traceback.print_exc()
                    stable = False
                    break
        return stable, max_diff

    def _init_feed_tensor(self, feed):
        for var in self.feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")

            var_in_scope = self.scope.find_var(var.name)
            assert var_in_scope, "Variable {} is not created.".format(var.name)
            tensor = var_in_scope.get_tensor()

            cur_feed = feed[var.name]
            if not isinstance(cur_feed, fluid.core.LoDTensor):
                tensor.set(cur_feed, self.place)
            else:
                raise TypeError("Feed data of non LoDTensor is not supported yet.")

    def _get_fetch_tensor(self):
        place = fluid.core.Place()
        place.set_place(fluid.CPUPlace())
        output = []
        for var in self.fetch_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Fetch data of non LoDTensor is not supported.")

            var_in_scope = self.scope.find_var(var.name)
            assert var_in_scope, "Variable {} is not created.".format(var.name)
            tensor = var_in_scope.get_tensor()

            cpu_tensor = tensor._copy(place)
            output.append(cpu_tensor)
        return output
