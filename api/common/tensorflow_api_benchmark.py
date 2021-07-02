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

import sys
import json
import time
import abc, six
import importlib
import numpy as np
from common import special_op_list

from . import utils
from . import api_param
from . import feeder

try:
    import tensorflow as tf
    from tensorflow.python.profiler import model_analyzer
    from tensorflow.python.profiler import option_builder
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.client import timeline
except Exception as e:
    sys.stderr.write(
        "Cannot import tensorflow, maybe tensorflow is not installed.\n")


class Profiler(object):
    def __init__(self, name, sess, profiler):
        self.name = name
        self.sess = sess
        self.profiler = profiler
        self.profiler_handle = None
        self.run_options = None
        self.run_metadata = None
        self.generate_timeline = False

    def __enter__(self):
        if self.profiler == "pyprof":
            import cProfile
            self.profiler_handle = cProfile.Profile()
            self.profiler_handle.enable()
        elif self.profiler != "none":
            self.profiler_handle = model_analyzer.Profiler(
                graph=self.sess.graph)
            if tf.__version__ < "1.15.0":
                self.run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata = tf.RunMetadata()
            else:
                self.run_options = tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                self.run_metadata = tf.compat.v1.RunMetadata()
        return self

    def add_step(self, step):
        if self.profiler != "none" and self.profiler != "pyprof":
            # Update profiler
            self.profiler_handle.add_step(
                step=step, run_meta=self.run_metadata)
            if self.generate_timeline:
                # For timeline
                tl = timeline.Timeline(self.run_metadata.step_stats)
                chrome_trace = tl.generate_chrome_trace_format()
                trace_file = open(self.name + '.tf.timeline', 'w')
                trace_file.write(chrome_trace)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.profiler == "pyprof":
            import pstats, StringIO
            self.profiler_handle.disable()
            # self.profiler_handle.dump_stats("./outputs/" + self.name + ".pyprof")
            s = StringIO.StringIO()
            ps = pstats.Stats(
                self.profiler_handle, stream=s).sort_stats("cumulative")
            ps.print_stats()
            print(s.getvalue())
        elif self.profiler != "none":
            # Generate profiling result
            profile_op_builder = option_builder.ProfileOptionBuilder().select(
                ['micros', 'occurrence']).order_by('micros').with_max_depth(5)
            self.profiler_handle.profile_operations(profile_op_builder.build())
        return self


def convert_dtype(dtype, to_string=True):
    def _trans(to_string, dtype_str, np_dtype):
        dtype = dtype_str if to_string else np.dtype(np_dtype)
        return dtype

    if dtype == tf.float16:
        # tf.float16: 16-bit half-precision floating-point.
        return _trans(to_string, "float16", np.float16)
    elif dtype == tf.float32:
        # tf.float32: 32-bit single-precision floating-point.
        return _trans(to_string, "float32", np.float32)
    elif dtype == tf.float64:
        # tf.float64: 64-bit double-precision floating-point.
        return _trans(to_string, "float64", np.float64)
    elif dtype == tf.int8:
        # tf.int8: 8-bit signed integer.
        return _trans(to_string, "int8", np.int8)
    elif dtype == tf.uint8:
        # tf.uint8: 8-bit unsigned integer.
        return _trans(to_string, "uint8", np.uint8)
    elif dtype == tf.uint16:
        # tf.uint16: 16-bit unsigned integer.
        return _trans(to_string, "uint16", np.uint16)
    elif dtype == tf.uint32:
        # tf.uint32: 32-bit unsigned integer.
        return _trans(to_string, "uint32", np.uint32)
    elif dtype == tf.uint64:
        # tf.uint64: 64-bit unsigned integer.
        return _trans(to_string, "uint64", np.uint64)
    elif dtype == tf.int16:
        # tf.int16: 16-bit signed integer.
        return _trans(to_string, "int16", np.int16)
    elif dtype == tf.int32:
        # tf.int32: 32-bit signed integer.
        return _trans(to_string, "int32", np.int32)
    elif dtype == tf.int64:
        # tf.int64: 64-bit signed integer.
        return _trans(to_string, "int64", np.int64)
    elif dtype == tf.bool:
        # tf.bool: Boolean.
        return _trans(to_string, "bool", np.bool)
    else:
        # tf.bfloat16: 16-bit truncated floating-point.
        # tf.complex64: 64-bit single-precision complex.
        # tf.complex128: 128-bit double-precision complex.
        # tf.string: String.
        # tf.qint8: Quantized 8-bit signed integer.
        # tf.quint8: Quantized 8-bit unsigned integer.
        # tf.qint16: Quantized 16-bit signed integer.
        # tf.quint16: Quantized 16-bit unsigned integer.
        # tf.qint32: Quantized 32-bit signed integer.
        # tf.resource: Handle to a mutable resource.
        # tf.variant: Values of arbitrary types.
        raise ValueError("Unsupported dtype %s" % dtype)


@six.add_metaclass(abc.ABCMeta)
class TensorflowAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.feed_list = None
        self.fetch_list = None
        self.allow_growth = True
        try:
            import tensorflow as tf
            self.graph = tf.Graph()
            if tf.__version__ > "1.15.0":
                tf.compat.v1.disable_eager_execution()
        except Exception as e:
            sys.stderr.write(
                "Cannot import tensorflow, maybe tensorflow is not installed.\n"
            )

    @abc.abstractmethod
    def build_graph(self, config=None):
        pass

    def placeholder(self, name, shape, dtype):
        tf_dtype = tf.as_dtype(dtype)
        if tf.__version__ >= "1.15.0":
            var = tf.compat.v1.placeholder(
                name=name, shape=shape, dtype=tf_dtype)
        else:
            var = tf.placeholder(name=name, shape=shape, dtype=tf_dtype)
        return var

    def variable(self, name, shape, dtype, value=None):
        assert shape is not None

        if self._feed_spec is not None and value is None:
            i = len(self._feed_dict)
            range = self._feed_spec[i].get("range", None)
        else:
            range = None
        feed_value = feeder.generate_random_data(
            shape, dtype, range=range, value=value)

        if self._need_feed:
            var = self.placeholder(name=name, shape=shape, dtype=dtype)
        else:
            var = tf.Variable(feed_value, name=name)

        if value is None:
            # When value is None, the variable is need to feed data.
            self._feed_dict[var] = feed_value
        return var

    def layers(self, api_name, module_name=None, **kwargs):
        def _import_func(tf_module_name, api_name):
            try:
                module = importlib.import_module(tf_module_name)
                func = getattr(module, api_name)
                print("Successly import %s.%s" % (tf_module_name, api_name))
                return func
            except Exception:
                print("Failed to import %s.%s" % (tf_module_name, api_name))
            return None

        tf_module_names = ["tensorflow", "tensorflow.math", "tensorflow.nn"]
        if module_name is not None and module_name not in tf_module_names:
            tf_module_names.append(module_name)

        for tf_module_name in tf_module_names:
            func = _import_func(tf_module_name, api_name)
            if func is not None:
                break

        assert func is not None, "Need to specify module_name to import %s." % api_name
        result = func(**kwargs)
        return result

    @property
    def backward(self):
        return self._backward

    def append_gradients(self, targets, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = tf.gradients(targets, inputs)
        self._backward = True
        print("Gradients: ", gradients)
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_list.append(grad)
        else:
            self.fetch_list.append(gradients)

    def _run_null_graph(self, use_gpu, repeat):
        walltimes = []
        graph = tf.Graph()
        with graph.as_default():
            x = tf.Variable(
                np.random.random([1]).astype("float32"), name="null")
            result = tf.identity(x)

            sess = self._init_session(use_gpu)
            for i in range(repeat + 1):
                begin = time.time()
                sess.run(fetches=[result.op], feed_dict=None)
                end = time.time()
                if i > 0:
                    walltimes.append(end - begin)
            sess.close()
        return walltimes

    def run_impl(self,
                 use_gpu,
                 feed,
                 repeat=1,
                 check_output=False,
                 profiler="none"):
        sess = self._init_session(use_gpu)

        #tf.debugging.set_log_device_placement(True)

        def _run_main_iter(run_options=None, run_metadata=None):
            feed_dict = feed if self._need_feed else None
            if self._need_fetch:
                fetches = self.fetch_list
            else:
                fetches = []
                for var in self.fetch_list:
                    fetches.append(var.op)
            outputs = sess.run(fetches=fetches,
                               feed_dict=feed_dict,
                               options=run_options,
                               run_metadata=run_metadata)
            return outputs

        if self.name != "null":
            walltimes = self._run_null_graph(use_gpu, repeat)

        # warmup run
        _run_main_iter(run_options=None, run_metadata=None)

        runtimes = []
        fetches = []
        outputs = None
        with Profiler(self.name, sess, profiler) as prof:
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter(
                    run_options=prof.run_options,
                    run_metadata=prof.run_metadata)
                runtimes.append(time.time() - begin)
                prof.add_step(step=i)

                if check_output:
                    fetches.append(outputs)
        sess.close()

        stats = {
            "framework": "tensorflow",
            "version": tf.__version__,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }
        if self.name != "null":
            stats["wall_time"] = walltimes
        return outputs, stats

    def generate_random_feeder(self,
                               config,
                               use_feed_fetch=True,
                               feeder_adapter=None):
        if config is None or not isinstance(config, api_param.APIConfig):
            raise ValueError(
                "Argument \"config\" must be set to an instance of APIConfig.")

        if feeder_adapter is not None and feeder_adapter.framework != "tensorflow":
            assert use_feed_fetch, "Argument use_feed_fetch must be True when feeder_adapter is initialized by paddle."

        if feeder_adapter is None or feeder_adapter.framework != "tensorflow":
            self._need_feed = config.name == "feed"
            self._need_fetch = use_feed_fetch or config.name == "fetch"
            self._feed_spec = feeder.copy_feed_spec(config.feed_spec)
            self._feed_dict = {}

            self._backward = False
            self.build_graph(config=config)

        if feeder_adapter is None:
            feed_list = []
            assert len(self._feed_dict) == len(self.feed_list)
            for var in self.feed_list:
                feed_list.append(self._feed_dict[var])
            return feeder.FeederAdapter("tensorflow", config.feed_spec,
                                        feed_list)
        else:
            return feeder_adapter

    def run(self, config, args, use_feed_fetch=True, feeder_adapter=None):
        self.name = config.api_name
        feeder_adapter = self.generate_random_feeder(config, use_feed_fetch,
                                                     feeder_adapter)
        if self._backward != args.backward:
            print(
                "Backward is not surported for %s in Tensorflow. It is actually running the forward test."
                % self.name)
            assert not special_op_list.has_backward(
                config
            ), "If backward is not surported for %s, please add the Paddle's \'%s\' to NO_BACKWARD_OPS in api/common/special_op_list.py." % (
                self.name, self.name)

        feed_list = feeder_adapter.to_tensorflow(self.feed_list)
        assert len(feed_list) == len(self.feed_list)
        feed = {}
        for i in range(len(feed_list)):
            feed[self.feed_list[i]] = feed_list[i]

        fetch_list = []
        for item in self.fetch_list:
            if isinstance(item, list):
                for var in item:
                    fetch_list.append(var)
            else:
                fetch_list.append(item)
        self.fetch_list = fetch_list

        self.allow_growth = False if args.task == "speed" else True
        outputs, stats = self.run_impl(
            use_gpu=args.use_gpu,
            feed=feed,
            repeat=args.repeat,
            check_output=args.check_output,
            profiler=args.profiler)
        return outputs, stats

    def _init_session(self, use_gpu):
        if tf.__version__ >= "1.15.0":
            config = tf.compat.v1.ConfigProto()
            if use_gpu:
                config.gpu_options.allow_growth = self.allow_growth
            else:
                # In default, TF use full cpu cores, but Paddle use one cpu core.
                # To make the same experiment, set TF use one cpu core as well.
                # See https://github.com/PaddlePaddle/Paddle/issues/18665#issuecomment-513780210
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
            sess = tf.compat.v1.Session(config=config)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = self.allow_growth
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        return sess
