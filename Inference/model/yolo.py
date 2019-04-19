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


class YoloBenchmark(benchmark.Benchmark):
    def __init__(self):
        """
        init
        """
        super(YoloBenchmark, self).__init__("yolo")

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
        #image_shspe
        im_shape_tensor = core.PaddleTensor()
        im_shape_tensor.shape = [1, 2]
        im_shape_tensor.name = "im_shape"
        im_shape_tensor.data = core.PaddleBuf([224, 224])
        im_shape_tensor.dtype = core.PaddleDType.INT32
        #im_id
        im_id_tensor = core.PaddleTensor()
        im_id_tensor.shape = [1, 1]
        im_id_tensor.name = "im_id"
        im_id_tensor.data = core.PaddleBuf([1,])
        im_id_tensor.dtype = core.PaddleDType.INT32
        return [image_tensor, im_shape_tensor]

