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


class TransformerBenchmark(benchmark.Benchmark):
    def __init__(self):
        """
        init
        """
        super(TransformerBenchmark, self).__init__("transformer")

    def get_input_data(self, file_path, batch_size=1):
        """
        get_input_data
        """
        num_lines = 0
        src_word, src_pos, src_slf_attn_bias, trg_word, init_score = [], [], [], [], []
        init_idx, trg_src_attn_bias, batch_data_shape = [], [], []
        with open(file_path) as file_handler:
            lines = file_handler.readlines()
            for line in lines:
                num_lines += 1
                data = line.strip("\n").split(",")
                temp_data = data[0].split(" ")
                src_word.append([int(temp) for temp in temp_data])
                temp_data = data[1].split(" ")
                src_pos.append([int(temp) for temp in temp_data])
                temp_data = data[2].split(" ")
                src_slf_attn_bias.append([float(temp) for temp in temp_data])
                temp_data = data[3].split(" ")
                trg_word.append([int(temp) for temp in temp_data])
                temp_data = data[4].split(" ")
                init_score.append([float(temp) for temp in temp_data])
                temp_data = data[5].split(" ")
                init_idx.append([int(temp) for temp in temp_data])
                temp_data = data[6].split(" ")
                trg_src_attn_bias.append([float(temp) for temp in temp_data])
                temp_data = data[7].split(" ")
                batch_data_shape.append([int(temp) for temp in temp_data])
                if num_lines == batch_size:
                    break
        src_word_data, src_pos_data, src_slf_attn_bias_data, \
                    trg_word_data = [], [], [], []
        init_score_data, init_idx_data, trg_src_attn_bias_data, \
                    batch_data_shape_data = [], [], [], []
        lod = [[], []]
        for i in range(batch_size):
            src_word_data.append(src_word[i])
            src_pos_data.append(src_pos[i])
            src_slf_attn_bias_data.append(src_slf_attn_bias[i])
            trg_word_data.append(trg_word[i])
            init_score_data.append(init_score[i])
            init_idx_data.append(init_idx[i])
            trg_src_attn_bias_data.append(trg_src_attn_bias[i])
        batch_data_shape_data.append(batch_data_shape[0])
        for i in range(batch_data_shape_data[0][0] + 1):
            lod[0].append(i)
            lod[1].append(i)
        return src_word_data, src_pos_data, src_slf_attn_bias_data, trg_word_data, \
                init_score_data, init_idx_data, trg_src_attn_bias_data, batch_data_shape_data, lod
    
    def load_data(self, filename):
        """
        load_data
        """
        src_word_data, src_pos_data, src_slf_attn_bias_data, trg_word_data, \
            init_score_data, init_idx_data, trg_src_attn_bias_data, \
            batch_data_shape_data, lod = self.get_input_data(filename)
        batch_size = batch_data_shape_data[0][0]
        n_head = batch_data_shape_data[0][1]
        trg_seq_len = batch_data_shape_data[0][2]
        src_seq_len = batch_data_shape_data[0][3]

        src_word_tensor = fluid.core.PaddleTensor()
        src_pos_tensor = fluid.core.PaddleTensor()
        src_slf_attn_bias_tensor = fluid.core.PaddleTensor()
        trg_word_tensor = fluid.core.PaddleTensor()
        init_score_tensor = fluid.core.PaddleTensor()
        init_idx_tensor = fluid.core.PaddleTensor()
        trg_src_attn_bias_tensor = fluid.core.PaddleTensor()

        src_word_tensor.name = "src_word"
        src_word_tensor.shape = (batch_size, src_seq_len, 1)
        src_word_tensor.dtype = fluid.core.PaddleDType.INT64
        the_data = []
        for data in src_word_data:
            the_data += data
        src_word_tensor.data = fluid.core.PaddleBuf(the_data)

        src_pos_tensor.name = "src_pos"
        src_pos_tensor.shape = (batch_size, src_seq_len, 1)
        src_pos_tensor.dtype = fluid.core.PaddleDType.INT64
        the_data = []
        for data in src_pos_data:
            the_data += data
        src_pos_tensor.data  = fluid.core.PaddleBuf(the_data)

        src_slf_attn_bias_tensor.name = "src_slf_attn_bias"
        src_slf_attn_bias_tensor.shape = (batch_size, n_head, src_seq_len, src_seq_len)
        src_slf_attn_bias_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        the_data = []
        for data in src_slf_attn_bias_data:
            the_data += data
        src_slf_attn_bias_tensor.data = fluid.core.PaddleBuf(the_data)

        trg_word_tensor.name = "trg_word"
        trg_word_tensor.shape = (batch_size, 1)
        trg_word_tensor.dtype = fluid.core.PaddleDType.INT64
        trg_word_tensor.lod = lod
        the_data = []
        for data in trg_word_data:
            the_data += data
        trg_word_tensor.data = fluid.core.PaddleBuf(the_data)

        init_score_tensor.name = "init_score"
        init_score_tensor.shape = (batch_size, 1)
        init_score_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        init_score_tensor.lod = lod
        the_data = []
        for data in init_score_data:
            the_data += data
        init_score_tensor.data = fluid.core.PaddleBuf(the_data)

        init_idx_tensor.name = "init_idx"
        init_idx_tensor.shape = (batch_size,)
        init_idx_tensor.dtype = fluid.core.PaddleDType.INT32
        the_data = []
        for data in init_idx_data:
            the_data += data
        init_idx_tensor.data = fluid.core.PaddleBuf(the_data)
    
        trg_src_attn_bias_tensor.name = "trg_src_attn_bias"
        trg_src_attn_bias_tensor.shape = (batch_size, n_head, trg_seq_len, src_seq_len)
        trg_src_attn_bias_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        the_data = []
        for data in trg_src_attn_bias_data:
            the_data += data
        trg_src_attn_bias_tensor.data = fluid.core.PaddleBuf(the_data)

        input = [src_word_tensor, src_pos_tensor, src_slf_attn_bias_tensor, trg_word_tensor, \
                init_score_tensor, init_idx_tensor, trg_src_attn_bias_tensor]
        return input




