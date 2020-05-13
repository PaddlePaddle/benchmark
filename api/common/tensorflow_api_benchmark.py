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
import numpy as np
import utils
import api_param
import feeder

try:
    import tensorflow as tf
    from tensorflow.python.profiler import model_analyzer
    from tensorflow.python.profiler import option_builder
    from tensorflow.python.client import timeline
except Exception as e:
    sys.stderr.write(
        "Cannot import tensorflow, maybe tensorflow is not installed.\n")


class Profiler(object):
    def __init__(self, name, sess, profile):
        self.name = name
        self.sess = sess
        self.profile = profile
        self.profiler = None
        self.run_options = None
        self.run_metadata = None
        self.generate_timeline = False

    def __enter__(self):
        if self.profile:
            self.profiler = model_analyzer.Profiler(graph=self.sess.graph)
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
        if self.profile:
            # Update profiler
            self.profiler.add_step(step=step, run_meta=self.run_metadata)
            if self.generate_timeline:
                # For timeline
                tl = timeline.Timeline(self.run_metadata.step_stats)
                chrome_trace = tl.generate_chrome_trace_format()
                trace_file = open(self.name + '.tf.timeline', 'w')
                trace_file.write(chrome_trace)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.profile:
            # Generate profiling result
            profile_op_builder = option_builder.ProfileOptionBuilder().select(
                ['micros', 'occurrence']).order_by('micros').with_max_depth(5)
            self.profiler.profile_operations(profile_op_builder.build())
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
        if self._use_feed_fetch:
            data = self.placeholder(name=name, shape=shape, dtype=dtype)
        else:
            assert value is not None
            assert isinstance(value, np.ndarray)
            value = feeder.check_shape_and_dtype(shape, dtype, value)
            data = tf.Variable(value, name=name)
        self.data = data
        return data

    def append_gradients(self, targets, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = [inputs]
        if not isinstance(inputs, list):
            raise TypeError("inputs should be a list.")

        gradients = tf.gradients(targets, inputs)
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_list.append(grad)
        else:
            self.fetch_list.append(gradients)

    def run_impl(self,
                 use_gpu,
                 feed=None,
                 repeat=1,
                 log_level=0,
                 check_output=False,
                 profile=False):
        sess = self._init_session(use_gpu)
        tf.debugging.set_log_device_placement(True)

        def _run_main_iter(feed=feed, run_options=None, run_metadata=None):
            if self._use_feed_fetch:
                fetches = self.fetch_list
            else:
                fetches = []
                for var in self.fetch_list:
                    fetches.append(var.op)
            outputs = sess.run(fetches=fetches,
                               feed_dict=feed,
                               options=run_options,
                               run_metadata=run_metadata)
            return outputs

        # warmup run
        _run_main_iter(feed=feed, run_options=None, run_metadata=None)

        runtimes = []
        fetches = []
        outputs = None
        with Profiler(self.name, sess, profile) as prof:
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter(
                    feed=feed,
                    run_options=prof.run_options,
                    run_metadata=prof.run_metadata)
                runtimes.append(time.time() - begin)
                prof.add_step(step=i)

                if check_output:
                    fetches.append(outputs)

        stats = {
            "framework": "tensorflow",
            "version": tf.__version__,
            "name": self.name,
            "total": runtimes
        }
        stats["device"] = "GPU" if use_gpu else "CPU"
        utils.print_benchmark_result(stats, log_level=log_level)
        return outputs

    def run(self, config, args, use_feed_fetch=True, feed_dict=None):
        if config is None or not isinstance(config, api_param.APIConfig):
            raise ValueError(
                "Argument \"config\" must be set to an instance of APIConfig.")

        self.name = config.name
        self._use_feed_fetch = use_feed_fetch
        if not use_feed_fetch:
            # For a test without feed and fetch, feeding data must be ready
            # before building graph and recorded in config.
            assert feed_dict is not None
            feed_dict = feeder.feed_tensorflow(
                feed_list=None,
                feed_dict_paddle=feed_dict,
                feed_spec=config.feed_spec)
            for name, value in feed_dict.items():
                setattr(config, name + "_data", value)
        else:
            for name, value in feed_dict.items():
                setattr(config, name + "_data", None)

        print(config)
        self.build_graph(config=config)
        if use_feed_fetch:
            feed_dict = feeder.feed_tensorflow(
                self.feed_list,
                feed_dict_paddle=feed_dict,
                feed_spec=config.feed_spec)
            assert len(feed_dict) == len(self.feed_list)

            feed = {}
            for var in self.feed_list:
                feed[var] = feed_dict[var.name]
        else:
            feed = None

        profile = True if args.profiler != "none" else False
        outputs = self.run_impl(
            use_gpu=args.use_gpu,
            feed=feed,
            repeat=args.repeat,
            log_level=args.log_level,
            check_output=args.check_output,
            profile=profile)
        return outputs

    def _init_session(self, use_gpu):
        if tf.__version__ >= "1.15.0":
            config = tf.compat.v1.ConfigProto()
            sess = tf.compat.v1.Session(config=config)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
        else:
            config = tf.ConfigProto()
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        return sess
