#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_name, log_file, res_log_file, device_num, fp_item):
    bs_pat = re.compile(r"samples_per_gpu=(.*),")
    time_pat = re.compile(r"(?<=, time: )\d+\.?\d*")
    loss_pat = re.compile(r"loss: (.*)")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    bs_res = bs_pat.findall(logs)
    time_res = time_pat.findall(logs)
    loss_res = loss_pat.findall(logs)

    print("---device_num:-", device_num)
    index_c = device_num.index('C')
    print("---index_c:-", index_c)
    gpu_num = int(device_num[index_c + 1:len(device_num)])
    print("-----gpu_num:", gpu_num)

    run_mode = ""
    # gpu_num = 0
    ips = 0
    bs = 0
    if time_res == []:
        ips = 0
    else:
        bs = int(bs_res[0])

        skip_num = 4
        total_time = 0
        for i in range(skip_num, len(time_res)):
            total_time += float(time_res[i])
        avg_time = total_time / (len(time_res) - skip_num)
        ips = round(bs * gpu_num / avg_time, 3)

    info = {    "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs,
                "fp_item": fp_item,
                "run_mode": "DP",
                "convergence_value": loss_res[-1],
                "convergence_key": "loss",
                "ips": ips,
                "speed_unit":"images/s",
                "device_num": device_num,
                "model_run_time": os.getenv('model_run_time'),
                "frame_commit": "",
                "frame_version": os.getenv('frame_version'),
        }
    json_info = json.dumps(info)
    print(json_info)
    with open(res_log_file, "w") as of:
        of.write(json_info)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    fp_item = sys.argv[5]
    analyze(model_item, log_file, res_log_file, device_num, fp_item)
