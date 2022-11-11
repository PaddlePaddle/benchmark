#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os

def analyze(model_item, log_file, res_log_file, device_num, fp_item, bs_item):
    time_pat = re.compile(r"train_samples_per_second = (.*)")
     
    logs = open(log_file).readlines()
    logs = ";".join(logs)
    time_res = time_pat.findall(logs)

    run_mode = "DP"
    gpu_num = 1
    ips = 0
    print("------time_res:", time_res[0])
    print("---device_num:-", device_num)
    index_c = device_num.index('C')
    print("---index_c:-", index_c)
    gpu_num = int(device_num[index_c + 1:len(device_num)])
    print("-----gpu_num:", gpu_num)    
 
    if time_res == []:
        ips = 0
    else:
        ips = round(float(time_res[0]), 3)
    model_name = model_item+"_"+"bs"+str(bs_item)+"_"+fp_item+"_"+run_mode
    info = {    "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs_item,
                "fp_item": fp_item,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": ips,
                "speed_unit":"sequences/s",
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
    fp_item  = sys.argv[5]
    bs_item = sys.argv[6]
    analyze(model_item, log_file, res_log_file, device_num, fp_item, bs_item)
