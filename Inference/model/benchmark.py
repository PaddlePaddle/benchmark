#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
from paddle import fluid

core = fluid.core
logger = logging.getLogger(__name__)


class Benchmark(object):
    def __init__(self,
                 name,
                 max_input_shape=None,
                 auto_config_layout=False,
                 passes_filter=None,
                 ops_filter=None):
        """
        __init__
        """
        self.name = name
        if max_input_shape is None:
            self.max_input_shape = {str(): list()}
        else:
            self.max_input_shape = max_input_shape
        self.auto_config_layout = auto_config_layout
        if passes_filter is None:
            self.passes_filter = list()
        else:
            self.passes_filter = passes_filter
        if ops_filter is None:
            self.ops_filter = list()
        else:
            self.ops_filter = ops_filter

    def set_config(self, **options):
        """
        set_config
        """
        use_gpu = options.get("use_gpu", False)
        use_tensorrt = options.get("use_tensorrt", False)
        use_anakin = options.get("use_anakin", False)
        gpu_memory = options.get("gpu_memory", 1000)
        #fraction_of_gpu_memory = options.get("fraction_of_gpu_memory", 0.9)
        device_id = options.get("device_id", 0)
        model_dir = options.get("model_dir")
        model_filename = options.get("model_filename")
        params_filename = options.get("params_filename")
        model_precision = options.get("model_precision")
        if model_filename and params_filename:
            prog_file = "%s/%s" % (model_dir, model_filename)
            params_file = "%s/%s" % (model_dir, params_filename)
            config = core.AnalysisConfig(prog_file, params_file)
        else:
            config = core.AnalysisConfig('benchmark')
            config.set_model(model_dir)
        #config.switch_ir_optim()
        #config.tensorrt_engine_enabled
        if use_gpu:
            config.enable_use_gpu(gpu_memory, device_id)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(8)
            config.enable_mkldnn()
        if use_tensorrt:
            print("use_tensorrt")
            if model_precision == "int8":
                print("use_tensorrt:int8")
                config.enable_tensorrt_engine(1 << 20, 1, precision_mode=core.AnalysisConfig.Precision.Int8)
            else:
                print("use_tensorrt:float")
                config.enable_tensorrt_engine(1 << 20, 1)
                #config.enable_tensorrt_engine(1 << 20, 1, min_subgraph_size=40)
        if use_anakin:
            print("use_anakin")
            if  model_precision == "int8":
                print("use_tensorrt:int8")
                config.enable_anakin_engine(max_batch_size=1,
                                            max_input_shape=self.max_input_shape,
                                            precision_mode=core.AnalysisConfig.Precision.Int8,
                                            auto_config_layout=self.auto_config_layout,
                                            passes_filter=self.passes_filter,
                                            ops_filter=self.ops_filter)
            else:
                print("use_tensorrt:float")
                config.enable_anakin_engine(max_batch_size=1,
                                            max_input_shape=self.max_input_shape,
                                            auto_config_layout=self.auto_config_layout,
                                            passes_filter=self.passes_filter,
                                            ops_filter=self.ops_filter)
        config.switch_ir_optim()
        self.config = config
        self.predictor = core.create_paddle_predictor(self.config)

    def load_data(self, **options):
        """
        load_data
        """
        raise NotImplementedError

    def run(self, tensor, warmup, repeat):
        """
        run
        """
        for i in range(warmup):
            output = self.predictor.run(tensor)
        t = []
        for i in range(repeat):
            t1 = time.time()
            output = self.predictor.run(tensor)
            t2 = time.time()
            t.append((t2 - t1)*1000)
        print("Run predictor for %d times, the average latenct is:%f ms" % (repeat, np.mean(t)))
