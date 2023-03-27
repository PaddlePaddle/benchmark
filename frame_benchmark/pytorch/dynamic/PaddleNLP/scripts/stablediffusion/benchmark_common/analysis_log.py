#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import sys
import json
import os
import re
from pathlib import Path
import numpy as np

def analyze(model_item, log_file, res_log_file, device_num, bs, fp_item, run_process_type):
    all_ips_value = []
    for p in Path(log_file).glob("**/workerlog*"):
        with open(str(p), "r", encoding="utf8") as f:
            data = f.readlines()
        ips_lines = []
        for eachline in data:
            if "ips:" in eachline:
                ips = float(eachline.split('ips: ')[1].split()[0])
                ips_lines.append(ips)
        all_ips_value.append(np.mean(ips_lines[2:42]))
    ips = round(np.sum(all_ips_value), 2)
    host_nums = int(re.findall("\d+",device_num)[0])
    ips *= host_nums
    run_mode = "DP"
    
    model_name = model_item+"_"+"bs"+str(bs)+"_"+fp_item+"_"+run_mode
    info = {
                "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs,
                "fp_item": fp_item,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": ips,
                "speed_unit": "text_image_pair/s",
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
    if len(sys.argv) != 8:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    bs  = int(sys.argv[5])
    fp_item  = sys.argv[6]
    run_process_type = sys.argv[7]
    
    analyze(model_item, log_file, res_log_file, device_num, bs, fp_item, run_process_type)
