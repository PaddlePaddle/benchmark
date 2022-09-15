#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import json
import os
import argparse


def analyze(args, run_info):
    log_file = args.filename
    res_log_file = args.res_log_file
    gpu_num = int(args.device_num[3:])
    if gpu_num == 32: 
        gpu_num = 8

    # caculate and update ips
    all_speed_logs =[]
    with open(log_file, 'r', encoding="utf8") as f:
        for line in f.readlines():
            if args.keyword in line:
                tokens_per_second = float(line.split("|")[5].replace("tokens/s", "").strip())
                all_speed_logs.append(tokens_per_second)

    # skip steps
    if args.skip_steps > 0 and len(all_speed_logs) > args.skip_steps:
        all_speed_logs = all_speed_logs[args.skip_steps: ]

    ips = sum(all_speed_logs) / len(all_speed_logs)
    run_info["ips"] = round(ips / gpu_num, 3)

    # write file
    run_info = json.dumps(run_info)
    print(run_info)
    with open(res_log_file, "w") as of:
        of.write(run_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    parser.add_argument(
        '--res_log_file', type=str, help='speed log file')
    parser.add_argument(
        '--skip_steps', type=int, default=10, help='The number of steps to be skipped')
    parser.add_argument(
        '--model_name', type=str, default=0, help='training model_name, transformer_base')
    parser.add_argument(
        '--device_num', type=str, default="N1C1", help='N1C1|N1C8|N4C32')
    parser.add_argument(
        '--run_process_type', type=str, default="SingleP", help='multi process or single process')
    parser.add_argument(
        "--keyword", type=str, default="|tokens/s", help="Keyword to specify analysis data")

    args = parser.parse_args()
    base_batch_size, fp_item, run_mode = args.model_name.split("_")[-3:]
    base_batch_size = int(base_batch_size.replace("bs",""))

    run_info = {    
                "model_branch": os.getenv('model_branch'),
                "model_commit": os.getenv('model_commit'),
                "model_name": args.model_name,
                "batch_size": base_batch_size,
                "fp_item": fp_item,
                "run_process_type": args.run_process_type,
                "run_mode": run_mode,
                "convergence_value": 0,
                "convergence_key": "",
                "ips": 0, # we need update ips
                "speed_unit": "tokens/s",
                "device_num": args.device_num,
                "model_run_time": os.getenv('model_run_time'),
                "frame_commit": "",
                "frame_version": os.getenv('frame_version'),
        }
    analyze(args, run_info)
