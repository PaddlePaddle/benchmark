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

import abc, six
import importlib


@six.add_metaclass(abc.ABCMeta)
class BenchmarkBase(object):
    def __init__(self, framework, testing_mode):
        self.name = self.__class__.__name__
        self._framework = framework
        self._testing_mode = testing_mode
        self._task = ""
        self.reset()

    def reset(self):
        if hasattr(self, "extra_tensors") and self.extra_tensors is not None:
            del self.extra_tensors
        self.extra_tensors = None
        self.feed_list = None
        self.fetch_list = None
        self._backward = False
        self._test_func = None
        self._test_kwargs = None

    @property
    def backward(self):
        return self._backward

    def compute_flop_and_byte(self, config):
        """ flop is used as a metric for op's performance and it is optional.
        """
        return None, None

    def build_graph(self, config=None):
        def _get_func(callable_api):
            callable_api_list = callable_api.split(".")
            func_name = callable_api_list[-1]
            callable_api_list.pop()
            module_name = ".".join(callable_api_list)
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                return func
            except Exception:
                print("Failed to import {}.{}".format(module_name, func_name))
            return None

        def _parse_api_signature(sig):
            # sig is like: "paddle.allclose(x, y, rtol, atol, equal_nan)"
            callable_api, args_str = sig.replace(" ", "").split("(")
            args = args_str.replace(")", "").split(",")
            return callable_api, args

        def _get_argument_name(args_dict, paddle_args, name):
            if self._framework == "paddle":
                assert name in paddle_args, "{} is expected to be in the argument list ({}).".format(
                    name, paddle_args)
                arg_name = name
            elif self._framework == "pytorch":
                arg_name = args_dict[name]
            return arg_name

        assert config is not None

        if self._test_func is None or self._test_kwargs is None:
            callable_api, paddle_args = _parse_api_signature(config.paddle_api)
            print("paddle: callable_api={}, args={}".format(callable_api,
                                                            paddle_args))
            if self._framework == "pytorch":
                callable_api, args = _parse_api_signature(config.torch_api)
                print("pytorch: callable_api={}, args={}".format(callable_api,
                                                                 args))
                assert len(paddle_args) == len(
                    args
                ), "The length of argument list of paddle and pytorch is expected to be the same, but recieved paddle ({}) vs pytorch ({}).".format(
                    paddle_args, args)
                args_dict = {}
                for i in range(len(args)):
                    args_dict[paddle_args[i]] = args[i]
            assert callable_api is not None

            self._test_func = _get_func(callable_api)
            self._test_kwargs = {}

            self.feed_list = []
            for var in config.variable_list:
                var_shape = getattr(config, var.name + '_shape')
                var_dtype = getattr(config, var.name + '_dtype')
                arg_name = _get_argument_name(args_dict, paddle_args, var.name)
                feed_var = self.variable(
                    name=var.name, shape=var_shape, dtype=var_dtype)
                self._test_kwargs[arg_name] = feed_var
                self.feed_list.append(feed_var)

            for param in config.params_list:
                arg_name = _get_argument_name(args_dict, paddle_args,
                                              param.name)
                self._test_kwargs[arg_name] = getattr(config, param.name)

        outputs = self._test_func(**self._test_kwargs)
        self.fetch_list = outputs if isinstance(outputs, list) else [outputs]

    @abc.abstractmethod
    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        pass

    @abc.abstractmethod
    def layers(self, api_name, module_name=None, **kwargs):
        pass

    @abc.abstractmethod
    def append_gradients(self, targets, inputs):
        pass

    def get_running_stats(self,
                          use_gpu,
                          config,
                          runtimes,
                          walltimes=None,
                          repeat=None):
        try:
            module_name = "torch" if self._framework == "pytorch" else self._framework
            module = importlib.import_module(module_name)
            version = module.__version__
        except Exception:
            version = "none"
            print("Failed to call %s.__version__" % (self._framework))

        stats = {
            "framework": self._framework,
            "version": version,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }

        if walltimes is not None:
            stats["wall_time"] = walltimes

        if repeat is not None:
            stats["repeat"] = repeat

        try:
            flop, byte = self.compute_flop_and_byte(config)
            if flop is not None:
                stats["flop"] = flop
            if byte is not None:
                stats["byte"] = byte
        except Exception:
            print("Failed to call compute_flops_and_byte for %s." %
                  (self._framework))

        return stats
