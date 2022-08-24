#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_item, log_file, res_log_file, device_num, batch_size, run_process_type):
    gpu_ids_pat = re.compile(r"rank (.*)")
    ips_pat = re.compile(r"avg_ips: (.*)")
    time_pat = re.compile(r"time: (.*), data_time")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    gpu_ids_res = gpu_ids_pat.findall(logs)
    ips_res = ips_pat.findall(logs)

    run_mode = ""
    gpu_num = 0
    ips = 0
    fp_item = "fp32"
    bs = batch_size

    if gpu_ids_res == []:
        ips = 0
    else:
        gpu_ids_res = [int(item) for item in gpu_ids_res]
        #gpu_num = int(max(gpu_ids_res) - min(gpu_ids_res) + 1)
        gpu_num = int(device_num[3:])
        print ("gpu_num",gpu_num)
        run_mode = "DP"

        skip_num = 4
        total_time = 0
        for i in range(skip_num, len(ips_res)):
            ips += float(ips_res[i])
        ips = ips / (len(ips_res) - skip_num)
    model_name = model_item+"_"+"bs"+str(bs)+"_"+fp_item+"_"+run_mode
    info = {    "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs,
                "fp_item": fp_item,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": ips * gpu_num,
                "speed_unit":"sentence/s",
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
    if len(sys.argv) != 7:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    batch_size = sys.argv[5]
    run_process_type = sys.argv[6]
    analyze(model_item, log_file, res_log_file, device_num, batch_size, run_process_type)
