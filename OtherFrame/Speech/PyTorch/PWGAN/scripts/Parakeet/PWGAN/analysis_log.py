#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import re
import sys
import json


def analyze(log_file, res_log_file,mission_name,direction_id):
    log_file_name = log_file.split("/")[-1]

    repo_name, model_name, run_mode, bs, fp, gpu_num = log_file_name.split("_")
    model_name = "_".join([repo_name, model_name, bs, fp])
    fail_flag = 0
    ips_res = []
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
        ips = ips * int(gpu_num)

        info = {"log_file": log_file, "model_name": model_name, "mission_name": mission_name,
                "direction_id": int(direction_id), "run_mode": run_mode, "index": 1, "gpu_num": int(gpu_num),
                "FINAL_RESULT": ips, "JOB_FAIL_FLAG": fail_flag, "UNIT": unit}
        json_info = json.dumps(info, ensure_ascii=False)
        with open(res_log_file, "w") as of:
            of.write(json_info)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:" + sys.argv[0] + "path/to/log/file path/to/res/log/file")
        sys.exit()

    log_file = sys.argv[1]
    res_log_file = sys.argv[2]
    mission_name = sys.argv[3]
    direction_id = sys.argv[4]

    analyze(log_file, res_log_file,mission_name,direction_id)
