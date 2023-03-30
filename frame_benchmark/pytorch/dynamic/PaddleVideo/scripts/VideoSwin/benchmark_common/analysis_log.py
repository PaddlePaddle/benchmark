#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import json
import os
import re
import sys


def analyze(model_item: str, log_file: str, res_log_file: str, device_num: int):
    """
    analyze raw log file and generate resolved file which saved at `res_log_file`
    log_file: such as 'temporal-shift-module_TSM_bs30_fp32_SingleP_DP_N1C1_log'
    res_log_file: such as 'temporal-shift-module_TSM_bs30_fp32_SingleP_DP_N1C1_speed'
    """
    ips_pat = re.compile(r"ips: \d+\.?\d+ instance/sec.")
    logs = open(log_file, "r").read().splitlines()
    logs = ";".join(logs)
    ips_res: list = ips_pat.findall(logs)

    log_file_name_split = os.path.basename(log_file).split("_")
    gpu_num = int(device_num[3:])
    run_mode = "DP"
    fp_item = log_file_name_split[-5]
    bs = int(log_file_name_split[2][2:])

    skip_num = 4
    if len(ips_res) > skip_num:
        ips_res = ips_res[skip_num:]
    model_name = model_item + "_" + "bs" + str(
        bs) + "_" + fp_item + "_" + run_mode
    ips_res = [
        float(ips.replace('ips: ', '').replace('instance/sec.', ''))
        for ips in ips_res
    ]
    avg_ips = sum(ips_res) / len(ips_res)
    info = {
        "model_branch": os.getenv('model_branch'),
        "model_commit": os.getenv('model_commit'),
        "model_name": model_name,
        "batch_size": bs,
        "fp_item": fp_item,
        "run_process_type": log_file_name_split[-4],
        "run_mode": run_mode,
        "convergence_value": 0,
        "convergence_key": '"loss":',
        "ips": avg_ips*gpu_num,
        "speed_unit": "instance/sec",
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
        print("Usage:" + sys.argv[0] +
              " model_item path/to/log/file path/to/res/log/file")
        sys.exit()

    model_item = sys.argv[1]  # name of model
    log_file = sys.argv[2]  # input raw log file path
    res_log_file = sys.argv[3]  # output resolved log file path
    device_num = sys.argv[4]  # GPU num
    analyze(model_item, log_file, res_log_file, device_num)  # run analyze.