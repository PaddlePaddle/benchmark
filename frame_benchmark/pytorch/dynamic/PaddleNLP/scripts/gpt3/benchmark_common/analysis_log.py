#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os


def analyze(model_item, log_file, res_log_file, device_num, bs, fp_item, run_process_type):
    time_pat = re.compile(r"elapsed time per iteration \(ms\): (.*) \| learning")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    time_res = time_pat.findall(logs)
    # print(gpu_ids_res)
    # print(bs_res)
    # print(time_res)
    # print(fp_res)
    run_mode = "DP"
    
    ips = 0
    try:
        # delete first 20 steps
        iter_times = [float(x) for x in  time_res[20:-1]]
        ips = bs * 1000 * 1024 /( sum(iter_times)/len(iter_times) )
    except Exception as e:
        print(e)
        ips = 0

    model_name = model_item+"_"+"bs"+str(bs)+"_"+fp_item+"_"+run_mode
    info = {
                "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": bs,
                "fp_item": fp_item,
                "run_process_type": run_process_type,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": ips,
                "speed_unit": "tokens/s",
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
