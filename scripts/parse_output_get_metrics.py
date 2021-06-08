#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# 针对动态图和静态图单独发报告

import os
import json
import argparse
import subprocess


diff_type = os.environ.get("diff_version_type")
if diff_type == "null":
    has_diff = False
else:
    has_diff = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--result_path",
    type=str,
    default='./result/new/static_graph',
    help="result dir path")
parser.add_argument(
    "--diff_path",
    type=str,
    default='./result/diff/static_graph',
    help="diff result dir path")
parser.add_argument(
    "--emails",
    type=str,
    default='',
    help="email receiver")
parser.add_argument(
    "--mode",
    type=str,
    default='static',
    help="benchmark mode")


class HTMLRender(object):
    """
    # 渲染生成 html 报告
    """

    def __init__(self, mode, env, result):
        """
        Args:
            mode(str): dynamic | static | dynamic_to_static
            env(dict): running environment, get config info from environment and commit id
            result(dict): html table data
        """
        self.mail_template = """From:paddle_benchmark@baidu.com
To:test@benchmark.com
Subject: TITLE_HOLDER
content-type:text/html
<html>
    <body>
        <h3 align=center>TITLE_HOLDER</h3>
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption>详细数据</caption>
            <tr bgcolor="#989898" >
TABLE_HEADER_HOLDER
            </tr>
TABLE_INFO_HOLDER
        </table>
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption bgcolor="#989898">环境配置</caption>
RUN_ENV_HOLDER
        </table>
        <HR align=center width="80%" SIZE=1>
    </body>
</html>
"""
        self.mode = mode
        self.title = "%s Benchmark 性能测试报告" % str(mode)
        self.mail_template = self.mail_template.replace("TITLE_HOLDER", self.title)
        self.env_content = ""
        self.table_header = ""
        self.table_info = ""
        self.__construct_env(env)
        self.__construct_table_info(result)

    def __construct_env(self, env):
        """
        # 渲染环境变量和配置信息，并在html表格中显示
        """
        if isinstance(env, dict):
            for k, v in env.items():
                self.env_content += """
                    <tr><td>{}</td><td>{}</td></tr>
                    """.format(k, v)
        self.mail_template = self.mail_template.replace("RUN_ENV_HOLDER", self.env_content)

    def __construct_table_info(self, result):
        """
        # 渲染 html 表格中详细数据
        """
        if has_diff:
            # 有diff数据
            if self.mode == "dynamic":
                headers_list = [
                    "方向", "模型",
                    "ips-单卡-本次", "ips-单卡-diff", "ips-单卡-波动比例",
                    "ips-八卡-本次", "ips-八卡-diff", "ips-八卡-波动比例",
                    "显存-单卡-本次", "显存-单卡-diff", "显存-单卡-波动比例",
                    "显存-八卡-本次", "显存-八卡-diff", "显存-八卡-波动比例",
                    "GPU利用率-单卡-本次", "GPU利用率-单卡-diff", "GPU利用率-单卡-波动比例",
                    "GPU利用率-八卡-本次", "GPU利用率-八卡-diff", "GPU利用率-八卡-波动比例",
                    "CPU利用率-单卡-本次", "CPU利用率-单卡-diff", "CPU利用率-单卡-波动比例",
                    "CPU利用率-八卡-本次", "CPU利用率-八卡-diff", "CPU利用率-八卡-波动比例",
                ]
                key_list = []
                for metric in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
                    for run_mode in ["sp", "mp"]:
                        for result_type in ["result", "diff", "percent"]:
                            if run_mode == "sp":
                                gpu_num = 1
                            else:
                                gpu_num = 8
                            key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
            else:
                headers_list = [
                    "方向", "模型",
                    "ips-单卡-本次", "ips-单卡-diff", "ips-单卡-波动比例",
                    "ips-单进程八卡-本次", "ips-单进程八卡-diff", "ips-单进程八卡-波动比例",
                    "ips-多进程八卡-本次", "ips-多进程八卡-diff", "ips-多进程八卡-波动比例",
                    "显存-单卡-本次", "显存-单卡-diff", "显存-单卡-波动比例",
                    "显存-单进程八卡-本次", "显存-单进程八卡-diff", "显存-单进程八卡-波动比例",
                    "显存-多进程八卡-本次", "显存-多进程八卡-diff", "显存-多进程八卡-波动比例",
                    "GPU利用率-单卡-本次", "GPU利用率-单卡-diff", "GPU利用率-单卡-波动比例",
                    "GPU利用率-单进程八卡-本次", "GPU利用率-单进程八卡-diff", "GPU利用率-单进程八卡-波动比例",
                    "GPU利用率-多进程八卡-本次", "GPU利用率-多进程八卡-diff", "GPU利用率-多进程八卡-波动比例",
                    "CPU利用率-单卡-本次", "CPU利用率-单卡-diff", "CPU利用率-单卡-波动比例",
                    "CPU利用率-单进程八卡-本次", "CPU利用率-单进程八卡-diff", "CPU利用率-单进程八卡-波动比例",
                    "CPU利用率-多进程八卡-本次", "CPU利用率-多进程八卡-diff", "CPU利用率-多进程八卡-波动比例",
                ]
                key_list = []
                for metric in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
                    for run_mode in ["sp", "mp"]:
                        if run_mode == "sp":
                            for gpu_num in [1, 8]:
                                for result_type in ["result", "diff", "percent"]:
                                    key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
                        else:
                            gpu_num = 8
                            for result_type in ["result", "diff", "percent"]:
                                key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
        else:
            # 无diff数据
            if self.mode == "dynamic":
                headers_list = [
                    "方向", "模型", "ips-单卡", "ips-八卡", "显存-单卡", "显存-八卡",
                    "GPU利用率-单卡", "GPU利用率-八卡", "CPU利用率-单卡", "CPU利用率-八卡",
                ]
                key_list = []
                for metric in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
                    for run_mode in ["sp", "mp"]:
                        result_type = "result"
                        if run_mode == "sp":
                            gpu_num = 1
                        else:
                            gpu_num = 8
                        key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
            else:
                headers_list = [
                    "方向", "模型", "ips-单卡", "ips-单进程八卡", "ips-多进程八卡",
                    "显存-单卡", "显存-单进程八卡", "显存-多进程八卡",
                    "GPU利用率-单卡", "GPU利用率-单进程八卡", "GPU利用率-多进程八卡",
                    "CPU利用率-单卡", "CPU利用率-单进程八卡", "CPU利用率-多进程八卡",
                ]
                key_list = []
                for metric in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
                    for run_mode in ["sp", "mp"]:
                        result_type = "result"
                        if run_mode == "sp":
                            for gpu_num in [1, 8]:
                                key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
                        else:
                            gpu_num = 8
                            key_list.append("%s_%s_%s_%s" % (result_type, run_mode, gpu_num, metric))
        for header_td in headers_list:
            self.table_header += """
                <td>{}</td>
                """.format(header_td)
        self.mail_template = self.mail_template.replace("TABLE_HEADER_HOLDER", self.table_header)
        for mission in result:
            for model_name in result[mission]:
                # 对应每一行
                self.table_info += "\t\t\t<tr><td bgcolor=white>%s</td><td bgcolor=white>%s</td>" % (
                    mission, model_name
                )
                for key in key_list:
                    # 对应每一个元素
                    try:
                        value = result[mission][model_name][key]
                    except Exception as e:
                        value = "---"
                    color = "white"
                    if "percent" in key:
                        try:
                            value = float(value)
                            if value > 5:
                                color = "green"
                            elif value < -5:
                                color = "red"
                            value = str(value) + "%"
                        except Exception as e:
                            pass
                    self.table_info += """
                    <td bgcolor={}>{}</td>
                    """.format(color, value)
                self.table_info += "</tr>\n"
        self.mail_template = self.mail_template.replace("TABLE_INFO_HOLDER", self.table_info)

    def save(self, html_file):
        """
        # 保存结果至指定html文件
        """
        with open(html_file, "w") as f:
            f.write(self.mail_template)


def _parameters_check(args):
    """
    # parameter check
    """
    print("result_path: %s" % args.result_path)
    print("diff_path: %s" % args.diff_path)
    print("emails: %s" % args.emails)
    print("mode: %s" % args.mode)
    result_path = os.path.abspath(args.result_path)
    diff_path = os.path.abspath(args.diff_path)
    mode = args.mode
    if not os.path.exists(result_path):
        raise Exception("result_path %s not exists" % str(result_path))
    if not os.path.exists(diff_path) and has_diff:
        raise Exception("diff_path %s not exists" % str(diff_path))
    if mode not in ["static", "dynamic"]:
        raise Exception("mode must in static or dynamic")


def _get_percent(result, mission, model_name, run_mode, gpu_num, key):
    """
    # 计算对应指标的变化率
    """
    try:
        new_value = result[mission][model_name]["result_%s_%s_%s" % (run_mode, gpu_num, key)]
        old_value = result[mission][model_name]["diff_%s_%s_%s" % (run_mode, gpu_num, key)]
        result = round((new_value - old_value) / old_value * 100, 3)
        return result
    except Exception as e:
        return "---"


def _parse_result_file(result_file):
    """
    # 解析 index 结果文件，返回结构化数据
    """
    try:
        gpu_memory = "---"
        gpu_used_avg = "---"
        cpu_used_avg = "---"
        commit_id = "----"
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("MAX_GPU_MEMORY_USE"):
                    gpu_memory = float(line.split("=")[-1].strip())
                if line.startswith("AVG_GPU_USE"):
                    gpu_used_avg = float(line.split("=")[-1].strip())
                if line.startswith("AVG_CPU_USE"):
                    cpu_used_avg = float(line.split("=")[-1].strip())
                if "Paddle commit is" in line:
                    commit_id = line.split("Paddle commit is")[-1].strip()
            file_line = lines[-1]
        json_info = json.loads(file_line)
        json_info["ips"] = json_info["FINAL_RESULT"]
        json_info["gpu_num"] = json_info["gpu_num"]
        json_info["gpu_memory"] = gpu_memory
        json_info["gpu_used_avg"] = gpu_used_avg
        json_info["cpu_used_avg"] = cpu_used_avg
        json_info["commit_id"] = commit_id
        return json_info
    except Exception as e:
        print("parse result file %s failed: %s" % (result_file, str(e)))
        return None


def _merge_result_info_into_result(result, json_info, result_type):
    """
    # 将单个模型的运行结果合入至整体result字典中
    result_type: result / diff
    """
    if json_info["mission_name"] not in result:
        result[json_info["mission_name"]] = {}
    if json_info["model_name"] not in result[json_info["mission_name"]]:
        result[json_info["mission_name"]][json_info["model_name"]] = {}
    for key in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
        result[json_info["mission_name"]][json_info["model_name"]][
            "%s_%s_%s_%s" % (result_type, json_info["run_mode"], json_info["gpu_num"], key)] = json_info[key]
    return result


def _calculate_percent_and_abnormal(result, mode):
    """
    # 遍历 result 中的每一个模型，补充相关的波动 percent 信息以及标记红、绿等颜色
    """
    for mission in result:
        for model_name in result[mission]:
            for run_mode in ["sp", "mp"]:
                for key in ["ips", "gpu_memory", "gpu_used_avg", "cpu_used_avg"]:
                    if mode == "static" and run_mode == "sp":
                        for gpu_num in [1, 8]:
                            metric_name = "percent_%s_%s_%s" % (run_mode, gpu_num, key)
                            result[mission][model_name][metric_name] = _get_percent(
                                result, mission, model_name, run_mode, gpu_num, key
                            )
                    elif run_mode == "mp":
                        gpu_num = 8
                        metric_name = "percent_%s_%s_%s" % (run_mode, gpu_num, key)
                        result[mission][model_name][metric_name] = _get_percent(
                            result, mission, model_name, run_mode, gpu_num, key
                        )
                    else:
                        gpu_num = 1
                        metric_name = "percent_%s_%s_%s" % (run_mode, gpu_num, key)
                        result[mission][model_name][metric_name] = _get_percent(
                            result, mission, model_name, run_mode, gpu_num, key
                        )


def _generate_and_send_html_report(result, emails, mode, env):
    """
    # 解析生成 html 报告并发送邮件
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file = current_dir + "/report.html"
    html_render = HTMLRender(mode, env, result)
    html_render.save(html_file)
    email_address = ",".join(x.strip() for x in emails.split(","))
    send_email_command = "cat %s |sendmail -t %s" % (html_file, email_address)
    subprocess.getstatusoutput(send_email_command)


def _process_run_log(args):
    """
    # process run result file
    """
    _parameters_check(args)
    result_path = os.path.abspath(args.result_path)
    diff_path = os.path.abspath(args.diff_path)
    result_listdir = sorted(os.listdir(result_path))
    result = {}
    new_commit_id = ""
    diff_commit_id = ""
    for file_name in result_listdir:
        result_info = _parse_result_file(result_path + "/" + file_name)
        if result_info:
            if not new_commit_id and result_info["commit_id"]:
                new_commit_id = result_info["commit_id"]
            result = _merge_result_info_into_result(result, result_info, "result")
    env = {
        "new_commit_id": new_commit_id,
        "test_version_type": os.environ.get("test_version_type"),
        "device_type": os.environ.get("device_type"),
        "cuda_version": os.environ.get("cuda_version"),
        "task_id": os.environ.get("task_id"),
        "diff_version_type": os.environ.get("diff_version_type")
    }
    test_version_type = os.environ.get("test_version_type")
    diff_version_type = os.environ.get("diff_version_type")
    if test_version_type == "pr":
        env["test_pr_number"] = os.environ.get("test_pr")
    elif test_version_type == "commit":
        env["test_commit_id"] = os.environ.get("test_commit_id")
    elif test_version_type == "branch":
        env["test_branch"] = os.environ.get("test_branch")
    if diff_version_type == "branch":
        env["diff_branch"] = os.environ.get("diff_branch")

    if has_diff:
        diff_listdir = sorted(os.listdir(diff_path))
        for file_name in diff_listdir:
            result_info = _parse_result_file(diff_path + "/" + file_name)
            if result_info:
                if not diff_commit_id and result_info["commit_id"]:
                    diff_commit_id = result_info["commit_id"]
                result = _merge_result_info_into_result(result, result_info, "diff")
        env["diff_commit_id"] = diff_commit_id
        _calculate_percent_and_abnormal(result, args.mode)
    _generate_and_send_html_report(result, args.emails, args.mode, env)


if __name__ == '__main__':
    args = parser.parse_args()
    _process_run_log(args)
