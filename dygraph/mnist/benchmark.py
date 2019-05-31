#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file benchmark.py
  * @author zhengtianyu
  * @date 2019/5/15 2:19 PM
  * @brief benchmark
  *
  **************************************************************************/
"""
import time


class Tools(object):
    def __init__(self):
        pass

    @staticmethod
    def time():
        """
        define time
        :return:
        """
        return time.time()

    @staticmethod
    def get_max_memory(file):
        """
        获取显存占用最大值
        :param file: 文件路径
        :return:
        """
        used_memory = set()
        # 舍弃前几行
        drop_line = 2
        with open(file, "r") as f:
            for i in range(drop_line):
                f.readline()

            for line in f.readlines():
                used = line.split()
                used_memory.add(int(used[2]))
        return max(list(used_memory))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        重置
        :return:
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新数值
        :param val:
        :param n:
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """
        重写str方法
        :return:
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    打印数据
    """
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        """
        重写print
        :param batch:
        :return:
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'