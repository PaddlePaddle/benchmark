#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_item, log_file, res_log_file, device_num, bs, fp_item, skip_num=2):
    time_pat = re.compile(r"time: (.*), data_time")
    logs = open(log_file).readlines()
    logs = ";".join(logs)
    time_res = time_pat.findall(logs)

    gpu_num = int(device_num[3:])
    run_mode = "DP"
    bs = int(bs)
    ips = 0

    
    if len(time_res) > skip_num:
        time_res = [float(a) for a in time_res]
        time_res = sorted(time_res)
        skip_num2 = max(int((len(time_res) * 0.05)), 5)
        time_res = time_res[skip_num2:len(time_res)-skip_num2]
        avg_time = sum(time_res) / len(time_res)
        ips = round(bs / avg_time, 3) * gpu_num
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
    if len(sys.argv) != 7:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    bs = sys.argv[5]
    fp_item = sys.argv[6]
    analyze(model_item, log_file, res_log_file, device_num, bs, fp_item)

