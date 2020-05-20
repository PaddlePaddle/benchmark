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
import importlib
import numpy as np
import utils

import api_param
import feeder

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

    @abc.abstractmethod
    def build_program(self, config=None):
        pass

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        assert shape is not None

        if self._feed_spec is not None and value is None:
            i = len(self._feed_dict)
            range = self._feed_spec[i].get("range", None)
        else:
            range = None
        feed_value = feeder.generate_random_data(
            shape, dtype, range=range, value=value)

        var = fluid.data(name=name, shape=shape, dtype=dtype, lod_level=0)
        var.persistable = True
        var.stop_gradient = stop_gradient

        if value is None:
            # When value is None, need to feed data to the variable.
            self._feed_dict[var] = feed_value
        return var

    def layers(self, name, **kwargs):
        module = importlib.import_module("paddle.fluid.layers")
        func = getattr(module, name)
        result = func(**kwargs)
        return result

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

    def _run_null_program(self, executor, repeat):
        walltimes = []
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(
                name="null", shape=[1], dtype="float32", lod_level=0)

            for i in range(repeat + 1):
                begin = time.time()
                outputs = executor.run(program=fluid.default_main_program(),
                                       feed=None,
                                       fetch_list=None,
                                       use_program_cache=True,
                                       return_numpy=True)
                end = time.time()
                if i > 0:
                    walltimes.append(end - begin)
        return walltimes

    def run_impl(self,
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
            if self._need_fetch:
                fetch_vars = self.fetch_vars
            else:
                fetch_vars = None
            outputs = executor.run(program=self.main_program,
                                   feed=feed,
                                   fetch_list=fetch_vars,
                                   use_program_cache=True,
                                   return_numpy=True)
            return outputs

        if self.name != "null":
            walltimes = self._run_null_program(executor, repeat)

        # warmup run
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
        if self.name != "null":
            stats["wall_time"] = walltimes
        stats["framework"] = "paddle"
        stats["version"] = paddle.__version__
        stats["name"] = self.name
        stats["device"] = "GPU" if use_gpu else "CPU"
        utils.print_benchmark_result(stats, log_level=log_level)
        return outputs

    def generate_random_feeder(self,
                               config,
                               use_feed_fetch=True,
                               feeder_adapter=None):
        if config is None or not isinstance(config, api_param.APIConfig):
            raise ValueError(
                "Argument \"config\" must be set to an instance of APIConfig.")

        if feeder_adapter is None or feeder_adapter.framework != "paddle":
            self._need_feed = use_feed_fetch or config.name == "feed"
            self._need_fetch = use_feed_fetch or config.name == "fetch"
            self._feed_spec = feeder.copy_feed_spec(config.feed_spec)
            self._feed_dict = {}

            self.main_program = fluid.Program()
            self.startup_program = fluid.Program()
            with fluid.program_guard(self.main_program, self.startup_program):
                self.build_program(config=config)

        if feeder_adapter is None:
            feed_list = []
            for var in self.feed_vars:
                feed_list.append(self._feed_dict[var])
            return feeder.FeederAdapter("paddle", config.feed_spec, feed_list)
        else:
            return feeder_adapter

    def run(self, config, args, use_feed_fetch=True, feeder_adapter=None):
        print(config)

        self.name = config.name
        feeder_adapter = self.generate_random_feeder(config, use_feed_fetch,
                                                     feeder_adapter)

        feed_list = feeder_adapter.to_paddle(self.feed_vars)
        assert len(feed_list) == len(self.feed_vars)
        if self._need_feed:
            feed = {}
            for i in range(len(feed_list)):
                feed[self.feed_vars[i].name] = feed_list[i]
        else:
            with fluid.program_guard(self.startup_program):
                # Append initialiar operator to startup program.
                for i in range(len(feed_list)):
                    self._assign(self.feed_vars[i], value=feed_list[i])
            feed = None

        outputs = self.run_impl(
            use_gpu=args.use_gpu,
            feed=feed,
            repeat=args.repeat,
            log_level=args.log_level,
            check_output=args.check_output,
            profiler=args.profiler)
        return outputs

    def _assign(self, feed_var, value):
        out = fluid.data(
            name=feed_var.name, shape=feed_var.shape, dtype=feed_var.dtype)
        out.persistable = True

        dtype_str = convert_dtype(feed_var.dtype)
        if dtype_str == "bool":
            value_name = "bool_values"
            value = [bool(v) for v in value.flat]
        elif dtype_str == "float32":
            value_name = "fp32_values"
            value = [float(v) for v in value.flat]
        elif dtype_str == "int32":
            value_name = "int32_values"
            value = [int(v) for v in value.flat]
        elif dtype == "int64":
            value_name = "int64_values"
            value = [int(v) for v in value.flat]
        else:
            raise TypeError(
                "The data type of 'value' must be bool, float32, int32 or int64, but "
                "received %s." % dtype_str)

        fluid.default_main_program().global_block().append_op(
            type='assign_value',
            outputs={'Out': [out]},
            attrs={
                'dtype': feed_var.dtype,
                'shape': list(feed_var.shape),
                value_name: value
            })

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
                raise TypeError(
                    "Feed data of non LoDTensor is not supported yet.")

    def _get_fetch_tensor(self):
        place = fluid.core.Place()
        place.set_place(fluid.CPUPlace())
        output = []
        for var in self.fetch_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError(
                    "Fetch data of non LoDTensor is not supported.")

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
            raise ValueError("The number of fetched results is {} (<= 0).".
                             format(len(fetches)))

        stable = True
        repeat = len(fetches)
        num_outputs = len(fetches[0])
        max_diff = 0.0
        for j in range(num_outputs):
            if not stable:
                break
            output_0 = None
            for i in range(repeat):
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
