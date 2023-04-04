#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_name, batch_size, log_file, res_log_file, device_num):
    gpu_ids_pat = re.compile(r"GPU (.*):")
    time_pat = re.compile(r"Batch time: (.*)\(.*\)")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    gpu_ids_res = gpu_ids_pat.findall(logs)
    time_res = time_pat.findall(logs)
    print(time_res, "***********************")

    print("---device_num:-", device_num)
    index_c = device_num.index('C')
    print("---index_c:-", index_c)
    gpu_num = int(device_num[index_c + 1:len(device_num)])
    print("-----gpu_num:", gpu_num)

    fail_flag = 0
    run_mode = ""
    fp_item = "fp32"
    ips = 0

    run_mode = "DP"
    skip_num = 10
    total_time = 0
    for i in range(skip_num, len(time_res)):
        total_time += float(time_res[i])
    avg_time = total_time / (len(time_res) - skip_num)
    ips = float(batch_size) * round(1 / avg_time, 3)

    info = {    "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": batch_size,
                "fp_item": fp_item,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": ips * int(gpu_num),
                "speed_unit":"images/s",
                "device_num": device_num,
                "model_run_time": os.getenv('model_run_time'),
                "frame_commit": "",
                "frame_version": os.getenv('frame_version'),
        }
    print(info)
    json_info = json.dumps(info)
    with open(res_log_file, "w") as of:
        of.write(json_info)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage:" + sys.argv[0] + " model_name path/to/log/file path/to/res/log/file")
        sys.exit()

    model_name = sys.argv[1]
    batch_size = sys.argv[2]
    log_file = sys.argv[3]
    res_log_file = sys.argv[4]
    device_num = sys.argv[5]

    analyze(model_name, batch_size, log_file, res_log_file, device_num) 
