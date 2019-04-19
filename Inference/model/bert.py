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


class BertBenchmark(benchmark.Benchmark):
    def __init__(self):
        """
        init
        """
        super(BertBenchmark, self).__init__("bert")

    def parse_line(self, line):
        """
        parse_line
        """
        tensors = []
        fields = line.strip("\n").split(";")
        #src_id
        src_id_tensor = self.parse_tensor(fields[0], "int")
        tensors.append(src_id_tensor)
        #pos_id
        pos_id_tensor = self.parse_tensor(fields[1], "int")
        tensors.append(pos_id_tensor)
        #segment_id
        segment_id_tensor = self.parse_tensor(fields[2], "int")
        tensors.append(segment_id_tensor)
        #self_attention_bias
        self_attention_bias_tensor = self.parse_tensor(fields[3], "float")
        tensors.append(self_attention_bias_tensor)
        return tensors

    def parse_tensor(self, field, data_type):
        """
        parse_tensor
        """
        tensor = fluid.core.PaddleTensor()
        data = field.split(":")
        temp_data = data[0].split(" ")
        shape = [int(temp) for temp in temp_data]
        temp_data = data[1].split(" ")
        tensor.shape = shape
        if data_type == "int":
            mat = [int(temp) for temp in temp_data]
            tensor.dtype = fluid.core.PaddleDType.INT64
        else:
            mat = [float(temp) for temp in temp_data]
            tensor.dtype = fluid.core.PaddleDType.FLOAT32
        tensor.data = fluid.core.PaddleBuf(mat)
        return tensor

    def load_data(self, filename, batch_size=1):
        """
        load_data
        """
        num_lines = 0
        inout_tensor = []
        with open(filename) as file_handler:
            lines = file_handler.readlines()
            for line in lines:
                num_lines += 1
                tensors = self.parse_line(line)
                inout_tensor += tensors
                if num_lines == batch_size:
                    break
        return inout_tensor




