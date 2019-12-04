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

import time
import abc, six
import traceback
import contextlib
import numpy as np
import paddle
import paddle.fluid as fluid
import utils


@contextlib.contextmanager
def profile_context(name, use_gpu, profiler):
    if profiler == "native":
        profile_type = "All" if use_gpu else "CPU"
        output_file = name + ".profile"
        with fluid.profiler.profiler(profile_type, 'total', output_file):
            yield
    elif profiler == "nvprof" and use_gpu:
        output_file = name + ".nvprof"
        with fluid.profiler.cuda_profiler(output_file, 'kvp'):
            yield
    else:
        yield


@six.add_metaclass(abc.ABCMeta)
class PaddleAPIBenchmarkBase(object):
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

    def append_gradients(self, targets, inputs):
        if isinstance(inputs, fluid.framework.Variable):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = fluid.backward.calc_gradient(targets, inputs)
        print(gradients)
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_vars.append(grad)
        else:
            self.fetch_vars.append(gradients)

    def run_with_executor(self, use_gpu, feed=None, repeat=1, log_level=0, check_output=False, profiler="none"):
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(self.place)
        executor.run(self.startup_program)

        if feed is None:
            feed = self._feed_random_data(use_gpu, as_lodtensor=True)

        runtimes = []
        fetches = []
        outputs = None
        with profile_context(self.name, use_gpu, profiler):
            for i in xrange(repeat):
                begin = time.time()
                outputs = executor.run(program=self.main_program,
                                      feed=feed,
                                      fetch_list=self.fetch_vars,
                                      use_program_cache=True,
                                      return_numpy=True)
                end = time.time()
                runtimes.append(end - begin)
                if check_output:
                    fetches.append(outputs)
        if check_output:
            stable, max_diff = self._check_consistency(fetches)
            stats = {"total": runtimes, "stable": stable, "diff": max_diff}
        else:
            stats = {"total": runtimes}
        stats["framework"] = "paddle"
        stats["version"] = paddle.__version__
        stats["name"] = self.name
        stats["device"] = "GPU" if use_gpu else "CPU"
        utils.print_stat(stats, log_level=log_level)
        return outputs

    def run_with_core_executor(self, use_gpu, feed=None, repeat=1, log_level=0, check_output=False, profiler="none"):
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
            feed = self._feed_random_data(use_gpu, as_lodtensor=False)

        feed_times = []
        fetch_times = []
        compute_times = []
        runtimes = []
        fetches = []
        outputs = None
        with profile_context(self.name, use_gpu, profiler):
            for i in xrange(repeat):
                begin = time.time()
                self._init_feed_tensor(feed)
                feed_end = time.time()
                core_executor.run_prepared_ctx(ctx, self.scope, False, False, False)
                compute_end = time.time()
                outputs = self._get_fetch_tensor()
                fetch_end = time.time()

                runtimes.append(fetch_end - begin)
                feed_times.append(feed_end - begin)
                compute_times.append(compute_end - feed_end)
                fetch_times.append(fetch_end - compute_end)
                
                if check_output:
                    fetches.append(outputs)
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
        stats["framework"] = "paddle"
        stats["version"] = paddle.__version__
        stats["name"] = self.name
        stats["device"] = "GPU" if use_gpu else "CPU"
        utils.print_stat(stats, log_level=log_level)
        return outputs

    def convert_dtype(self, dtype, to_string=True):
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

    def _feed_random_data(self, use_gpu, as_lodtensor=False):
        print("feed random data")
        feed = {}
        if use_gpu and as_lodtensor:
            #place = fluid.CPUPlace()
            place = fluid.CUDAPinnedPlace()
        for var in self.feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")
                
            shape = var.shape
            dtype = self.convert_dtype(var.dtype, to_string=True)
            data = np.random.random(shape).astype(dtype)
            if use_gpu and as_lodtensor:
                tensor = fluid.core.LoDTensor()
                tensor.set(data, place)
                feed[var.name] = tensor
            else:
                feed[var.name] = data
        return feed

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
                    diff = utils.compare(output_0, output_i)
                    max_diff = diff if diff > max_diff else max_diff
                except (RuntimeError, ValueError, AssertionError) as e:
                    traceback.print_exc()
                    stable = False
                    break
        return stable, max_diff
