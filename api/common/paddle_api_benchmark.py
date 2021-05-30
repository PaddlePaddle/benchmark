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
import logging
import warnings
import numpy as np
import sys

from common import utils
from common import api_param
from common import feeder
from common import special_op_list

try:
    import paddle
    import paddle.fluid as fluid
except Exception as e:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")

try:
    paddle.enable_static()
except Exception:
    print(
        "The paddle version is less than 2.0, it can not use paddle.enable_static()"
    )


@contextlib.contextmanager
def profile_context(name, use_gpu, profiler):
    if profiler in ["Default", "OpDetail", "AllOpDetail"]:
        profile_type = "All" if use_gpu else "CPU"
        output_file = "./outputs/" + name + ".pd.profile"
        with fluid.profiler.profiler(
                profile_type, 'total', output_file, tracer_option=profiler):
            yield
    elif profiler == "pyprof":
        import cProfile, pstats
        from io import StringIO

        profiler_handle = cProfile.Profile()
        profiler_handle.enable()
        yield
        profiler_handle.disable()
        # profiler_handle.dump_stats("./outputs/" + name + ".pyprof")
        s = StringIO()
        ps = pstats.Stats(profiler_handle, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
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
        self.scope = None
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

    def fluid_layers(self, name, **kwargs):
        module = importlib.import_module("paddle.fluid.layers")
        func = getattr(module, name)
        result = func(**kwargs)
        return result

    def layers(self, api_name, module_name=None, **kwargs):
        def _import_func(paddle_module_name, api_name):
            try:
                module = importlib.import_module(paddle_module_name)
                func = getattr(module, api_name)
                print("Successly import %s.%s" %
                      (paddle_module_name, api_name))
                return func
            except Exception:
                print("Failed to import %s.%s" %
                      (paddle_module_name, api_name))
            return None

        paddle_module_names = ["paddle", "paddle.nn.functional"]
        if module_name is not None and module_name not in paddle_module_names:
            paddle_module_names.append(module_name)

        for paddle_module_name in paddle_module_names:
            func = _import_func(paddle_module_name, api_name)
            if func is not None:
                break

        assert func is not None, "Need to specify module_name to import %s." % api_name
        result = func(**kwargs)
        return result

    @property
    def backward(self):
        return self._backward

    def append_gradients(self, targets, inputs):
        if isinstance(inputs, fluid.framework.Variable):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = fluid.backward.gradients(targets, inputs)
        self._backward = True
        print("Gradients: ", gradients)
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
                 feed,
                 repeat=1,
                 check_output=False,
                 profiler="none"):
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        executor.run(self.startup_program)

        stats = {
            "framework": "paddle",
            "version": paddle.__version__,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward
        }

        def _run_main_iter():
            feed_dict = feed if self._need_feed else None
            fetch_vars = self.fetch_vars if self._need_fetch else None
            outputs = executor.run(program=self.main_program,
                                   feed=feed_dict,
                                   fetch_list=fetch_vars,
                                   use_program_cache=True,
                                   return_numpy=True)
            if use_gpu:
                paddle.fluid._cuda_synchronize(paddle.fluid.CUDAPlace(0))
            return outputs

        if self.name != "null":
            walltimes = self._run_null_program(executor, repeat)

        if not self._need_feed:
            self._init_feed_tensor(use_gpu, feed)

        try:
            # warmup run
            outputs = _run_main_iter()

            runtimes = []
            fetches = []
            outputs = None
            with profile_context(self.name, use_gpu, profiler):
                for i in range(repeat):
                    begin = time.time()
                    outputs = _run_main_iter()
                    runtimes.append(time.time() - begin)
                    if check_output:
                        fetches.append(outputs)

            stats["total"] = runtimes
            if check_output:
                stable, max_diff = self._check_consistency(fetches)
                stats["stable"] = stable
                stats["diff"] = max_diff
            if self.name != "null":
                stats["wall_time"] = walltimes
            return outputs, stats
        except fluid.core.EnforceNotMet as ex:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.error(ex.message)
            return False, stats

    def generate_random_feeder(self,
                               config,
                               use_feed_fetch=True,
                               feeder_adapter=None):
        if config is None or not isinstance(config, api_param.APIConfig):
            raise ValueError(
                "Argument \"config\" must be set to an instance of APIConfig.")

        if feeder_adapter is None or feeder_adapter.framework != "paddle":
            self._need_feed = config.name == "feed"
            self._need_fetch = use_feed_fetch or config.name == "fetch"
            self._feed_spec = feeder.copy_feed_spec(config.feed_spec)
            self._feed_dict = {}

            self._backward = False
            self.main_program = fluid.Program()
            self.startup_program = fluid.Program()
            with fluid.program_guard(self.main_program, self.startup_program):
                self.build_program(config=config)

            # For backward benchmark, the program is composed of:
            #   xxx -> shape -> fill_constant -> xxx_grad
            # The extra CUDA kernel of fill_constant will make the traced times
            # larger than the actual, but tf can automatic optimize the execution
            # of fill_constant. We call self._prune() to move the fill_constant op
            # from main_program to startup_program for current benchmark and will
            # optimize the execution strategy in the future.
            self._prune(config)

        if feeder_adapter is None:
            feed_list = []
            for var in self.feed_vars:
                feed_list.append(self._feed_dict[var])
            return feeder.FeederAdapter("paddle", config.feed_spec, feed_list)
        else:
            return feeder_adapter

    def run(self, config, args, use_feed_fetch=True, feeder_adapter=None):
        self.name = config.api_name
        feeder_adapter = self.generate_random_feeder(config, use_feed_fetch,
                                                     feeder_adapter)
        if self._backward != args.backward:
            print(
                "Backward is not surported for %s in Paddle. It is actually running the forward test."
                % self.name)
            assert not special_op_list.has_backward(
                config
            ), "If backward is not surported for %s, please add the \'%s\' to NO_BACKWARD_OPS in api/common/special_op_list.py." % (
                self.name, self.name)

        feed_list = feeder_adapter.to_paddle(self.feed_vars)
        assert len(feed_list) == len(self.feed_vars)
        feed = {}
        for i in range(len(feed_list)):
            feed[self.feed_vars[i].name] = feed_list[i]

        self.scope = fluid.Scope()
        with fluid.scope_guard(self.scope):
            outputs, stats = self.run_impl(
                use_gpu=args.use_gpu,
                feed=feed,
                repeat=args.repeat,
                check_output=args.check_output,
                profiler=args.profiler)
        return outputs, stats

    def _init_feed_tensor(self, use_gpu, feed):
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        for var in self.feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")

            var_in_scope = self.scope.var(var.name)
            tensor = var_in_scope.get_tensor()

            cur_feed = feed[var.name]
            if isinstance(cur_feed, np.ndarray) or isinstance(
                    cur_feed, fluid.core.LoDTensor):
                tensor.set(cur_feed, place)
            else:
                raise TypeError(
                    "Feed data of non LoDTensor/np.ndarray is not supported yet."
                )

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

    def _prune(self, config):
        if not config.backward or config.api_name in [
                "while_loop", "case", "switch_case"
        ]:
            return

        main_block = self.main_program.global_block()
        startup_block = self.startup_program.global_block()

        index = None
        for i in range(len(main_block.ops) - 1):
            if main_block.ops[i].type == "shape" and main_block.ops[
                    i + 1].type == "fill_constant":
                index = i
                break
        if index is None:
            return

        shape_op = main_block.ops[index]
        fill_constant_op = main_block.ops[index + 1]
        target_var = main_block.var(shape_op.input("Input")[0])
        target_grad_var = main_block.var(fill_constant_op.output("Out")[0])
        if -1 in target_var.shape:
            return

        dtype = target_grad_var.dtype
        attrs = {
            "shape": target_var.shape,
            "value": 1.0,
            "dtype": target_var.dtype
        }
        target_grad_var_copy = startup_block.create_var(
            name=target_grad_var.name,
            dtype=target_grad_var.dtype,
            persistable=True)
        startup_block.append_op(
            type="fill_constant",
            inputs=None,
            outputs={"Out": [target_grad_var_copy]},
            attrs=attrs,
            stop_gradient=True)
        target_grad_var.persistable = True

        main_block._remove_var(shape_op.output("Out")[0])
        main_block._remove_op(index + 1)
        main_block._remove_op(index)

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
