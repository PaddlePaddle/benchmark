#!/bin/python
# -*- coding: UTF-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import time
import json
import op_benchmark_unit
import pymysql
import paddle
import torch
import platform
import socket
from datetime import datetime

COMPARE_RESULT_SHOWS = {
    "Better": "Better",
    "Equal": "Equal",
    "Less": "Less",
    "Unknown": "Unknown",
    "Unsupport": "Unsupport",
    "Others": "Others",
    "Total": "Total"
}


class DB(object):
    """
    DB class
    dump data to mysql database
    """

    def __init__(self):
        self.db = pymysql.connect(
            # 手动填写内容
            host='xxxxxx',  # 数据库地址
            user='xxxxxx',  # 数据库用户名
            password='******',  # 数据库密码
            db='xxxxxx',  # 数据库名称
            port=3306,
            use_unicode=True,
            charset="utf8"
            # charset = 'utf8 -- UTF-8 Unicode'
        )
        self.cursor = self.db.cursor()

    def timestamp(self):
        """
        时间戳控制
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def init_mission(self):
        """init mission"""
        sql = (
            "insert into `jobs` (`create_time`, `update_time`, `status`, `paddle_commit`, "
            "`paddle_version`, `torch_version`, `mode`, `hostname`,`system`) "
            "values ('{}','{}','{}', '{}', '{}', '{}', '{}', '{}', '{}' );".
            format(
                self.timestamp(),
                self.timestamp(),
                "running",
                paddle.__git_commit__,
                paddle.__version__,
                torch.__version__,
                "op-benchmark",
                socket.gethostname(),
                platform.platform(), ))
        try:
            self.cursor.execute(sql)
            self.job_id = self.db.insert_id()
            self.db.commit()
        except Exception as e:
            print(e)

    def save(self, benchmark_result_list, compare_framework=None):
        """更新job状态"""
        retry = 3
        for i in range(retry):
            sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
                self.timestamp(), "saving", self.job_id)
            try:
                self.cursor.execute(sql)
                self.db.commit()
                break
            except Exception:
                # 防止超时失联
                self.db.ping(True)
                continue
        # 插入数据

        json_result = {"paddle": dict(), "pytorch": dict(), "compare": dict()}

        for device in ["cpu", "gpu"]:
            json_result["device"] = device
            for case_id in range(len(benchmark_result_list)):
                op_unit = benchmark_result_list[case_id]
                for direction in ["forward", "backward", "backward_forward"]:
                    result = op_unit.get(device, direction)
                    time_set = ["total",
                                "gpu_time"] if device == "gpu" else ["total"]
                    for key in time_set:
                        compare_result = COMPARE_RESULT_SHOWS.get(
                            result["compare"][key], "--")
                        json_result["compare"][direction +
                                               key] = compare_result
                        for framework in ["paddle", compare_framework]:
                            json_result[framework][
                                "case_name"] = op_unit.case_name
                            json_result[framework][direction + key] = result[
                                framework][key]
                sql = (
                    "insert into `cases`(`jid`, `case_name`, `result`, `create_time`) "
                    "values ('{}', '{}', '{}', '{}')".format(
                        self.job_id, json_result[framework]["case_name"],
                        json.dumps(json_result), self.timestamp()))
                print(type(sql))
                try:
                    self.cursor.execute(sql)
                    self.db.commit()
                except Exception as e:
                    print(e)

        # 更新job状态
        sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
            self.timestamp(), "done", self.job_id)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(e)

    def error(self):
        """错误配置"""
        retry = 3
        for i in range(retry):
            sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
                self.timestamp(), "error", self.job_id)
            try:
                self.cursor.execute(sql)
                self.db.commit()
                break
            except Exception as e:
                # 防止超时失联
                self.db.ping(True)
                print(e)
                continue
