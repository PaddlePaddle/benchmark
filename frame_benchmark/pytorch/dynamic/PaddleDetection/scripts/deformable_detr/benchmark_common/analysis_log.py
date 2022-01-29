#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_item, log_file, res_log_file, device_num):
    gpu_ids_pat = re.compile(r"gpu_ids = range(.*)")
    bs_pat = re.compile(r"samples_per_gpu=(.*),")
    time_pat = re.compile(r"time: (.*), data_time")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    gpu_ids_res = gpu_ids_pat.findall(logs)
    bs_res = bs_pat.findall(logs)
    time_res = time_pat.findall(logs)

    run_mode = ""
    gpu_num = 0
    ips = 0
    fp_item = "fp32"
    bs = 0

    if gpu_ids_res == [] or time_res == []:
        ips = 0
    else:
        gpu_num = int(gpu_ids_res[0][4]) - int(gpu_ids_res[0][1])
        bs = int(bs_res[0])
        run_mode = "SP" if gpu_num == 1 else "MP"

        skip_num = 4
        total_time = 0
        for i in range(skip_num, len(time_res)):
            total_time += float(time_res[i])
        avg_time = total_time / (len(time_res) - skip_num)
        ips = round(bs / avg_time, 3)
    model_name = model_item+"_"+"bs"+str(bs)+"_"+fp_item+"_"+run_mode
    info = {    "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs,
                "fp_item": fp_item,
                "run_process_type": "MultiP",
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
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
    if len(sys.argv) != 5:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    analyze(model_item, log_file, res_log_file, device_num)
