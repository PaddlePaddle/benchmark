#!/usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================

"""
@Desc: post_log module
@File: post_log.py
@Author: liangjinhua
@Date: 2020/07/21 15:22
"""
import requests
import argparse
import os
import sys
from requests.adapters import HTTPAdapter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--file_path",
    type=str,
    default="",
    help="The path of log file. (default: %(default)d)")

parser.add_argument(
    "--server_path",
    type=str,
    default="",
    help="The path of http server(default: %(default)d)")

if __name__ == "__main__":
    args = parser.parse_args()
    print("file_path is : {}".format(args.file_path))
    print("http server is : {}".format(args.server_path))
    file_list = []
    if os.path.isdir(args.file_path):
        for dir_path, dir_names, file_names in os.walk(args.file_path):
            for file_name in file_names:
                file_list.append({
                                       'abs_path': os.path.join(dir_path, file_name),
                                       'file_path': 'op_benchmark/{}'.format(
                                           os.path.basename(args.file_path)
                                       )
                                       })
    elif os.path.exists(args.file_path):
        file_list.append(args.file_path)
    else:
        print("no file")
        sys.exit(0)
    print("file_path: {} contains: {} files".format(args.file_path, len(file_list)))
    with requests.Session() as s:
        s.mount('http://', HTTPAdapter(max_retries=3))
        s.mount('https://', HTTPAdapter(max_retries=3))
        for file_info in file_list:
            data = {'file_path': file_info['file_path']}
            files = {'file_name': open(file_info['abs_path'], 'rb')}
            r = s.post(args.server_path, data, files=files)
