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
import numpy as np
import paddle.fluid as fluid

class APIBenchmarkBase(object):
    __metaclass__ = abc.ABCMeta      

    def __init__(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.scope = fluid.Scope()
        self.feed_vars = None
        self.fetch_vars = None

    @abc.abstractmethod
    def build_program(self, backward=False):
        pass

    def run_with_executor(self, use_gpu, feed=None, repeat=1, log_level=0):
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        executor.run(self.startup_program)

        if feed is None:
            feed = self._feed_random_data()

        runtimes = []
        for i in xrange(repeat):
            begin = time.time()
            executor.run(program=self.main_program,
                         feed=feed,
                         fetch_list=self.fetch_vars,
                         use_program_cache=True,
                         return_numpy=True)
            end = time.time()
            runtimes.append(end - begin)
        self._print_stat(runtimes, log_level=log_level)

    def run_with_core_executor(self, use_gpu, feed=None, repeat=1):
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        executor.run(self.startup_program)

        # Use to run main_program
        core_executor = fluid.core.Executor(place)

        fetch_list_str = list(map(_to_name_str, self.fetch_list))
        ctx = core_executor.prepare(
                    self.main_program.desc, 0, fetch_list_str, False)
        core_executor.create_variables(self.main_program.desc, self.scope, 0)
 
        for i in xrange(repeat):
            core_executor.run_prepared_ctx(ctx, self.scope, False, False, False)

    def _print_stat(self, runtimes, log_level=0):
        if not isinstance(runtimes, list):
            raise TypeError("runtimes is not a list.")

        for i in xrange(len(runtimes)):
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
        avg_time = np.average(sorted_runtimes[begin:end])

        if log_level == 1:
            for i in xrange(len(runtimes)):
                print("Iter {0}, Runtime: {1}".format("%4d" % i, "%.5f ms" % runtimes[i]))

        print("Total {0}, Analysis range [{1}, {2}), Average: {3}".format(len(sorted_runtimes), begin, end, "%.5f ms" % avg_time))

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

    def _feed_random_data(self):
        print("feed random data")
        feed = {}
        for var in self.feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")
                
            shape = var.shape
            dtype = self._convert_dtype(var.dtype, to_string=True)
            data = np.random.random(shape).astype(dtype)
            feed[var.name] = data
        return feed

    def _fetch_data(self):
        pass
