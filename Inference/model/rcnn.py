#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import matplotlib.image

from paddle import fluid
from .reader import process_image
import benchmark

core = fluid.core
logger = logging.getLogger(__name__)


class RcnnBenchmark(benchmark.Benchmark):
    def __init__(self):
        """
        init
        """
        name = "rcnn"
        max_input_shape = {"data": [1, 3, 1333, 1333], "im_info": [1, 3]}
        auto_config_layout = True
        passes_filter = ["fc_fuse_pass"]
        ops_filter = ["mul", "fc", "softmax", "reshape2"]
        super(RcnnBenchmark, self).__init__(name,
                                            max_input_shape,
                                            auto_config_layout,
                                            passes_filter,
                                            ops_filter)

    def load_data(self, filename):
        """
        load_data
        """
        image = process_image(filename, 224)
        image_tensor = core.PaddleTensor()
        image_tensor.name = 'image'
        image_tensor.shape = [1, 3, 224, 224]
        image_tensor.data = core.PaddleBuf(image.tolist())
        image_tensor.dtype = core.PaddleDType.FLOAT32
        #im-info
        im_info_tensor = core.PaddleTensor()
        im_info_tensor.shape = [1, 3]
        im_info_tensor.name = "im_info"
        im_info_tensor.data = core.PaddleBuf([224., 224., 1.0])
        im_info_tensor.dtype = core.PaddleDType.FLOAT32
        return [image_tensor, im_info_tensor]

