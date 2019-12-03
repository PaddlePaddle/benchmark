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

import json
import time
import abc, six
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.client import timeline
import utils


@six.add_metaclass(abc.ABCMeta)
class TensorflowAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.feed_list = None
        self.fetch_list = None
        self.allow_growth = False

    @abc.abstractmethod
    def build_graph(self, backward=False):
        pass

    def run(self, use_gpu, feed=None, repeat=1, log_level=0, check_output=False, profile=False):
        config = self._set_config(use_gpu)
        if tf.__version__ < "1.14.0":
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:
            sess = tf.compat.v1.Session(config=config)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

        if profile:
            profiler = model_analyzer.Profiler(graph=sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            log_writer = None
            run_options = None
            run_metadata = None
        self.timeline_dict = None

        if feed is None:
            feed = self._feed_random_data()

        runtimes = []
        fetches = []
        outputs = None
        for i in xrange(repeat):
            begin = time.time()
            outputs = sess.run(fetches=self.fetch_list,
                               feed_dict=feed,
                               options=run_options,
                               run_metadata=run_metadata)
            end = time.time()
            runtimes.append(end - begin)

            if profile:
                # Update profiler
                profiler.add_step(step=i, run_meta=run_metadata)
                # For timeline
                tl = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = tl.generate_chrome_trace_format()
                trace_file = open(self.name + '_tf.timeline', 'w')
                trace_file.write(chrome_trace)
                #self._update_timeline(chrome_trace)

            if check_output:
                fetches.append(outputs)
        if profile:
            # Generate profiling result
            profile_op_builder = option_builder.ProfileOptionBuilder()
            profile_op_builder.select(['micros','occurrence'])
            profile_op_builder.order_by('micros')
            profile_op_builder.with_max_depth(10)
            profiler.profile_operations(profile_op_builder.build())
            # Generate timeline
#            profile_graph_builder = option_builder.ProfileOptionBuilder(
#                                    option_builder.ProfileOptionBuilder.time_and_memory())
#            profile_graph_builder.with_timeline_output(timeline_file=self.name + '_tf.timeline')
#            profile_graph_builder.with_step(10)
#            profiler.profile_graph(profile_graph_builder.build())
            #tl_output_file = self.name + "_tf.timeline"
            #with open(tl_output_file, 'w') as f:
            #    json.dump(self.timeline_dict, f)

        stats = {"framework": "tensorflow", "version": tf.__version__, "name": self.name, "total": runtimes}
        stats["device"] = "GPU" if use_gpu else "CPU"
        utils.print_stat(stats, log_level=log_level)
        return outputs

    def _set_config(self, use_gpu):
        if tf.__version__ < "1.14.0":
            config = tf.ConfigProto()
        else:
            config = tf.compat.v1.ConfigProto()

#        if use_gpu:
#            if not self.allow_growth:
#                config.gpu_options.per_process_gpu_memory_fraction = 0.9
#            else:
#                config.gpu_options.allow_growth = True
            #config.log_device_placement = True
        return config

    def _update_timeline(self, chrome_trace):
        # Codes from: https://github.com/ikhlestov/tensorflow_profiling/blob/master/03_merged_timeline_example.py
        # Convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        if self.timeline_dict is None:
            # For first run store full trace
            self.timeline_dict = chrome_trace_dict
        else:
            # For other - update only time consumption, not definitions
            for event in chrome_trace_dict['traceEvents']:
                # Events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self.timeline_dict['traceEvents'].append(event)

    def convert_dtype(self, dtype, to_string=True):
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

    def _feed_random_data(self):
        print("feed random data")
        feed = {}
        for var in self.feed_list:
            shape = var.shape
            dtype = self.convert_dtype(var.dtype, to_string=True)
            feed[var] = np.random.random(shape).astype(dtype)
        return feed
