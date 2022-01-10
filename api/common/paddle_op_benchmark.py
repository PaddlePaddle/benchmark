#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import json
import time
import importlib
import numpy as np

from common import utils
from common import api_param
from common import feeder
from common import special_op_list
from common.benchmark import BenchmarkBase
from common.paddle_api_benchmark import profile_context

try:
    import paddle
except Exception as e:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")


class StaticHelper(object):
    def __init__(self):
        self._feed_spec = None
        self._feed_dict = {}

    def set_feed_spec(self, feed_spec):
        self._feed_spec = feeder.copy_feed_spec(feed_spec)

    def set_feed_dict(self, feed_dict=None):
        if feed_dict is None:
            self._feed_dict = {}
        else:
            self._feed_dict = feed_dict

    def get_feed_dict(self):
        return self._feed_dict

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        assert shape is not None

        if self._feed_spec is not None and value is None:
            i = len(self._feed_dict)
            range = self._feed_spec[i].get("range", None)
        else:
            range = None
        feed_value = feeder.generate_random_data(
            shape, dtype, range=range, value=value)

        var = paddle.static.data(
            name=name, shape=shape, dtype=dtype, lod_level=0)
        var.persistable = True
        var.stop_gradient = stop_gradient

        if value is None:
            # When value is None, need to feed data to the variable.
            self._feed_dict[var] = feed_value
        return var

    def generate_gradients(self, targets, inputs):
        if isinstance(inputs, paddle.static.Variable):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = paddle.static.gradients(targets, inputs)
        return gradients

    def init_feed_tensor(self, use_gpu, feed_vars, feed_dict, scope):
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        for var in feed_vars:
            if var.type != paddle.fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")

            var_in_scope = scope.var(var.name)
            tensor = var_in_scope.get_tensor()

            cur_feed = feed_dict[var.name]
            if isinstance(cur_feed, np.ndarray) or isinstance(
                    cur_feed, paddle.fluid.core.LoDTensor):
                tensor.set(cur_feed, place)
            else:
                raise TypeError(
                    "Feed data of non LoDTensor/np.ndarray is not supported yet."
                )

    def get_fetch_tensor(self, fetch_vars, scope):
        place = paddle.fluid.core.Place()
        place.set_place(paddle.CPUPlace())
        output = []
        for var in fetch_vars:
            if var.type != paddle.fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError(
                    "Fetch data of non LoDTensor is not supported.")

            var_in_scope = scope.find_var(var.name)
            assert var_in_scope, "Variable {} is not created.".format(var.name)
            tensor = var_in_scope.get_tensor()

            cpu_tensor = tensor._copy(place)
            output.append(cpu_tensor)
        return output

    def run_null_program(self, executor, repeat):
        walltimes = []
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.static.data(
                name="null", shape=[1], dtype="float32", lod_level=0)

            for i in range(repeat + 1):
                begin = time.time()
                outputs = executor.run(
                    program=paddle.static.default_main_program(),
                    feed=None,
                    fetch_list=None,
                    use_program_cache=True,
                    return_numpy=True)
                end = time.time()
                if i > 0:
                    walltimes.append(end - begin)
        return walltimes

    def prune(self, config, main_program, startup_program):
        if not config.backward or config.api_name in [
                "while_loop", "case", "switch_case"
        ]:
            return

        main_block = main_program.global_block()
        startup_block = startup_program.global_block()

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


class DynamicHelper(object):
    BEFORE_RUN = 0
    IN_RUN = 1
    AFTER_RUN = 2

    def __init__(self):
        self._feed_spec = None
        self._feed_values = None
        self._feed_dict = {}
        self._ones_like_targets = []
        self._status = DynamicHelper.BEFORE_RUN

    def set_feed_spec(self, feed_spec):
        self._feed_spec = feeder.copy_feed_spec(feed_spec)

    def set_feed_values(self, feed_values):
        self._feed_values = feed_values

    def set_feed_dict(self, feed_dict=None):
        if feed_dict is None:
            self._feed_dict = {}
        else:
            self._feed_dict = feed_dict

    def switch_status(self, status=None):
        if status is not None:
            self._status = status
        else:
            self._status = (self._status + 1) % 3

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        if self._status == DynamicHelper.BEFORE_RUN:
            if self._feed_values is not None and value is None:
                i = len(self._feed_dict)
                feed_value = feeder.check_shape_and_dtype(
                    shape=shape, dtype=dtype, value=self._feed_values[i])
            else:
                assert shape is not None

                if self._feed_spec is not None and value is None:
                    i = len(self._feed_dict)
                    range = self._feed_spec[i].get("range", None)
                else:
                    range = None
                feed_value = feeder.generate_random_data(
                    shape, dtype, range=range, value=value)
            var = paddle.to_tensor(feed_value, stop_gradient=stop_gradient)
            self._feed_dict[name] = var
        else:
            var = self._feed_dict[name]
        return var

    def generate_gradients(self, targets, inputs):
        if not isinstance(targets, list):
            if len(self._ones_like_targets) == 0:
                ones_like_targets = paddle.ones_like(targets)
                self._ones_like_targets.append(ones_like_targets)
            else:
                ones_like_targets = self._ones_like_targets[0]
        else:
            ones_like_targets = None
        gradients = paddle.grad(
            outputs=targets, inputs=inputs, grad_outputs=ones_like_targets)
        return gradients


class PaddleOpBenchmarkBase(BenchmarkBase):
    def __init__(self, testing_mode):
        super(PaddleOpBenchmarkBase, self).__init__(testing_mode)

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        return self._helper.variable(name, shape, dtype, value, stop_gradient)

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

        if self._layers_function is None:
            paddle_module_names = ["paddle", "paddle.nn.functional"]
            if module_name is not None and module_name not in paddle_module_names:
                paddle_module_names.append(module_name)

            for paddle_module_name in paddle_module_names:
                func = _import_func(paddle_module_name, api_name)
                if func is not None:
                    break

            assert func is not None, "Need to specify module_name to import %s." % api_name
            self._layers_function = func

        result = self._layers_function(**kwargs)
        return result

    def append_gradients(self, targets, inputs):
        gradients = self._helper.generate_gradients(targets, inputs)
        self._backward = True
        # print("Gradients: ", gradients)
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_list.append(grad)
        else:
            self.fetch_list.append(gradients)

    def run(self, config, args, use_feed_fetch=True, feeder_adapter=None):
        self._layers_function = None
        if self._testing_mode == "dynamic":
            self._helper = DynamicHelper()
            return self._run_dynamic(config, args, feeder_adapter)
        elif self._testing_mode == "static":
            self._helper = StaticHelper()
            return self._run_static(config, args, use_feed_fetch,
                                    feeder_adapter)
        else:
            return None, None

    def _run_dynamic_impl(self,
                          use_gpu,
                          config,
                          repeat=1,
                          profiler="none",
                          feeder_adapter=None):
        assert self._testing_mode == "dynamic", "Function \"_run_dynamic_impl\" can only be called when self._testing_mode is dynamic, but recieved {}.".format(
            self._testing_mode)

        def _run_main_iter():
            self.build_graph(config=config)
            if use_gpu:
                paddle.device.cuda.synchronize(0)

            outputs = None
            if self._need_fetch:
                outputs = []
                for var in self.fetch_list:
                    if isinstance(var, np.ndarray):
                        outputs.append(var)
                    else:
                        outputs.append(var.numpy())
            return outputs

        # warmup run
        _run_main_iter()

        runtimes = []

        self._helper.switch_status()
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter()
                runtimes.append(time.time() - begin)

        self._helper.switch_status()
        stats = self._get_output_stats(use_gpu, config, runtimes)
        return outputs, stats

    def _run_dynamic(self, config, args, feeder_adapter=None):
        assert self._testing_mode == "dynamic", "Function \"_run_dynamic\" can only be called when self._testing_mode is dynamic, but recieved {}.".format(
            self._testing_mode)

        paddle.disable_static()
        self.name = config.api_name

        self._need_fetch = args.task == "accuracy"
        self._helper.set_feed_spec(config.feed_spec)
        self._helper.set_feed_dict(feed_dict={})
        self._helper.switch_status(status=DynamicHelper.BEFORE_RUN)
        if feeder_adapter:
            self._helper.set_feed_values(feeder_adapter.to_paddle())
        outputs, stats = self._run_dynamic_impl(
            use_gpu=args.use_gpu,
            config=config,
            repeat=args.repeat,
            profiler=args.profiler,
            feeder_adapter=feeder_adapter)
        return outputs, stats

    def _run_static_impl(self,
                         use_gpu,
                         config,
                         feed,
                         repeat=1,
                         profiler="none"):
        assert self._testing_mode == "static", "Function \"_run_static_impl\" can only be called when self._testing_mode is static, but recieved {}.".format(
            self._testing_mode)

        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        executor = paddle.static.Executor(place)
        executor.run(self.startup_program)

        def _run_main_iter():
            feed_dict = feed if self._need_feed else None
            fetch_vars = self.fetch_list if self._need_fetch else None
            outputs = executor.run(program=self.main_program,
                                   feed=feed_dict,
                                   fetch_list=fetch_vars,
                                   use_program_cache=True,
                                   return_numpy=True)
            if use_gpu:
                paddle.device.cuda.synchronize(0)
            return outputs

        if self.name != "null":
            walltimes = self._helper.run_null_program(executor, repeat)

        if not self._need_feed:
            self._helper.init_feed_tensor(use_gpu, self.feed_list, feed,
                                          self.scope)

        try:
            # warmup run
            outputs = _run_main_iter()

            runtimes = []
            outputs = None
            with profile_context(self.name, use_gpu, profiler):
                for i in range(repeat):
                    begin = time.time()
                    outputs = _run_main_iter()
                    runtimes.append(time.time() - begin)

            stats = self._get_output_stats(use_gpu, config, runtimes, walltimes
                                           if self.name != "null" else None)
            return outputs, stats
        except paddle.fluid.core.EnforceNotMet as ex:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.error(ex.message)
            return False, stats

    def generate_random_feeder(self,
                               config,
                               use_feed_fetch=True,
                               feeder_adapter=None):
        assert self._testing_mode == "static", "Function \"generate_random_feeder\" can only be called when self._testing_mode is static, but recieved {}.".format(
            self._testing_mode)

        if config is None or not isinstance(config, api_param.APIConfig):
            raise ValueError(
                "Argument \"config\" must be set to an instance of APIConfig.")

        if feeder_adapter is None or feeder_adapter.framework != "paddle":
            self._need_feed = config.name == "feed"
            self._need_fetch = use_feed_fetch or config.name == "fetch"
            self._helper.set_feed_spec(config.feed_spec)
            self._helper.set_feed_dict(feed_dict={})

            self._backward = False
            self.main_program = paddle.static.Program()
            self.startup_program = paddle.static.Program()
            with paddle.static.program_guard(self.main_program,
                                             self.startup_program):
                self.build_graph(config=config)

            # For backward benchmark, the program is composed of:
            #   xxx -> shape -> fill_constant -> xxx_grad
            # The extra CUDA kernel of fill_constant will make the traced times
            # larger than the actual, but tf can automatic optimize the execution
            # of fill_constant. We call self._prune() to move the fill_constant op
            # from main_program to startup_program for current benchmark and will
            # optimize the execution strategy in the future.
            self._helper.prune(config, self.main_program, self.startup_program)

        if feeder_adapter is None:
            feed_list = []
            for var in self.feed_list:
                feed_list.append(self._helper.get_feed_dict()[var])
            return feeder.FeederAdapter("paddle", config.feed_spec, feed_list)
        else:
            return feeder_adapter

    def _run_static(self,
                    config,
                    args,
                    use_feed_fetch=True,
                    feeder_adapter=None):
        assert self._testing_mode == "static", "Function \"_run_static\" can only be called when self._testing_mode is static, but recieved {}.".format(
            self._testing_mode)

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

        feed_list = feeder_adapter.to_paddle(self.feed_list)
        assert len(feed_list) == len(self.feed_list)
        feed = {}
        for i in range(len(feed_list)):
            feed[self.feed_list[i].name] = feed_list[i]

        self.scope = paddle.static.Scope()
        with paddle.static.scope_guard(self.scope):
            outputs, stats = self._run_static_impl(
                use_gpu=args.use_gpu,
                config=config,
                feed=feed,
                repeat=args.repeat,
                profiler=args.profiler)
        return outputs, stats

    def _get_output_stats(self, use_gpu, config, runtimes, walltimes=None):
        stats = {
            "framework": "paddle",
            "version": paddle.__version__,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }

        if walltimes is not None:
            stats["wall_time"] = walltimes

        flop, byte = self.compute_flop_and_byte(config)
        if flop is not None:
            stats["flop"] = flop
        if byte is not None:
            stats["byte"] = byte
        return stats
