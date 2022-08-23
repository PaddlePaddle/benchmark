#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os


def analyze(model_item, log_file, res_log_file, device_num):
    bs_pat = re.compile(r"global_batch_size ............................... (.*)")
    time_pat = re.compile(r"elapsed time per iteration \(ms\): (.*) \| learning")
    gpu_ids_pat = re.compile(r"world_size ......................................(.*)")
    fp_pat = re.compile(r"fp16 ............................................(.*)")

    logs = open(log_file).readlines()
    logs = ";".join(logs)
    bs_res = bs_pat.findall(logs)
    time_res = time_pat.findall(logs)
    gpu_ids_res = gpu_ids_pat.findall(logs)
    fp_res = fp_pat.findall(logs)
    # print(gpu_ids_res)
    # print(bs_res)
    # print(time_res)
    # print(fp_res)

    run_mode = ""
    gpu_num = 0
    ips = 0
    fp_item = "fp32"
    bs = 0
    run_process_type = "SingleP" 

    if gpu_ids_res == [] or time_res == []:
        ips = 0
    else:
        # delete first 20 steps
        iter_times = [float(x) for x in  time_res[20:-1]]
        gpu_num = int(gpu_ids_res[-1])
        run_mode = "DP"
        run_process_type = "SingleP" if gpu_num == 1 else "MultiP"
        if fp_res[-1].strip().lower() == "true":
            fp_item = "fp16" 
        bs = int(bs_res[-1])
        ips = bs * 1000 * 1024 /( sum(iter_times)/len(iter_times) )

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
    if len(sys.argv) != 5:
        print("Usage:" + sys.argv[0] + " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    analyze(model_item, log_file, res_log_file, device_num)
