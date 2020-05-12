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
import utils

try:
    import paddle
    import paddle.fluid as fluid
except Exception as e:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")


@contextlib.contextmanager
def profile_context(name, use_gpu, profiler):
    if profiler in ["Default", "OpDetail", "AllOpDetail"]:
        profile_type = "All" if use_gpu else "CPU"
        output_file = "./outputs/" + name + ".pd.profile"
        with fluid.profiler.profiler(
                profile_type, 'total', output_file, tracer_option=profiler):
            yield
    elif profiler == "nvprof" and use_gpu:
        output_file = name + ".nvprof"
        with fluid.profiler.cuda_profiler(output_file, 'kvp'):
            yield
    else:
        yield


def convert_dtype(dtype, to_string=True):
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


@six.add_metaclass(abc.ABCMeta)
class PaddleAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.scope = fluid.Scope()
        self.place = None
        self.feed_vars = None
        self.fetch_vars = None
        self.feed_tensors = {}

    @abc.abstractmethod
    def build_program(self, config=None):
        pass

    def create_program(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

    def append_gradients(self, targets, inputs):
        if isinstance(inputs, fluid.framework.Variable):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = fluid.backward.gradients(targets, inputs)
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_vars.append(grad)
        else:
            self.fetch_vars.append(gradients)

    def run(self,
            use_gpu,
            feed=None,
            repeat=1,
            log_level=0,
            check_output=False,
            profiler="none"):
        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(self.place)
        executor.run(self.startup_program)

        def _run_main_iter(feed=None):
            outputs = executor.run(program=self.main_program,
                                   feed=feed,
                                   fetch_list=self.fetch_vars,
                                   use_program_cache=True,
                                   return_numpy=True)
            return outputs

        # warmup, filling the feeding data.
        outputs = _run_main_iter(feed=feed)

        runtimes = []
        fetches = []
        outputs = None
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter(feed=feed)
                runtimes.append(time.time() - begin)
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
            raise ValueError("The number of fetched results is {} (<= 0).".
                             format(len(fetches)))

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
