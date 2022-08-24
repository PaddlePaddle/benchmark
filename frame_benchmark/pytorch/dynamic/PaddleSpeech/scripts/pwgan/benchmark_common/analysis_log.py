#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json
import os


def analyze(model_name, log_file, res_log_file, device_num):
    log_file_name = log_file.split("/")[-1]
    # model_name: pwgan_bs6_fp32_SingleP_DP
    model_item, bs, fp_item, run_process_type, run_mode = "DP"
    gpu_num = 0
    ips = 0
    fail_flag = 0
    ips_res = []
    gpu_num = int(device_num[3:])

    print("gpu_num:", gpu_num)

    with open(log_file, "r") as rf:
        for line in rf:
            line = line.strip()
            if "avg_ips" in line:
                line_ips = line.split(":")[-1].strip()
                ips_res.append(line_ips)
    if ips_res == []:
        fail_flag = 1
    else:
        unit = ips_res[0].split(" ")[-1]
        skip_num = 4
        clip_ips_res = ips_res[skip_num:]
        clip_ips_res = [float(item.split(" ")[0]) for item in clip_ips_res]
        ips = sum(clip_ips_res) / len(clip_ips_res)

        info = {"model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": model_name,
                "batch_size": int(bs[2:]),
                "fp_item": fp_item,
                "run_process_type": run_process_type,
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
        json_info = json.dumps(info, ensure_ascii=False)
        with open(res_log_file, "w") as of:
            of.write(json_info)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:" + sys.argv[0] + " model_name path/to/log/file path/to/res/log/file")
        sys.exit()

    model_name = sys.argv[1]
    log_file = sys.argv[2]
    res_log_file = sys.argv[3]
    device_num = sys.argv[4]
    analyze(model_name, log_file, res_log_file, device_num)
