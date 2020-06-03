#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================

"""
@Desc: template module
@File: template.py
@Author: liangjinhua
@Date: 2019/5/30 19:26
"""
import os
import save

MAIL_HEAD_CONTENT = """
From:paddle_benchmark@baidu.com
To:test@benchmark.com
Subject:benchmark运行结果报警，请检查
content-type:text/html
<html>
    <body>
        <h3 align=center>benchmark alarm email</h3>
        <HR align=center width="80%" SIZE=1>         
        <table border="1" align=center>
        <caption bgcolor="#989898">环境配置</caption>
RUN_ENV_HOLDER 
        </table>            
"""

MAIL_MID_CONTENT = """
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption>报警结果_INDEX_HOLDER</caption>
            <tr bgcolor="#989898" ><td>模型</td><td>运行环境</td><td>指标</td><td>标准值</td><td>当前值</td><td>波动范围</td>JOB_LINK_HOLDER</tr>
ALARM_INFO_HOLDER
        </table>
"""

MAIL_TAIL_CONTENT = """
        <HR align=center width="80%" SIZE=1>
        <h4 align=center>历史详细数据 BENCHMARK_WEBSITE</h4> 
    </body>
</html>
"""


def construct_mid_content(results, single=True):
    """the list not satify condition
    Args:
        results(dict): {1:[[], []], 2:[[], []]}
        single(bool):
    returns:
        alarm_info_holder(str)
    """
    def get_alarm_info(alarm_list):
        """construnct alarm info for each restult.
        Args: alarm_list(list):[[],[]]
        """
        alarm_info = ""
        for each_res in alarm_list:
            if not each_res:
                continue
            range_index = len(each_res) - 1 if single else len(each_res) - 2
            alarm_info += "            <tr>"
            index_type = each_res[2]
            for i in range(len(each_res)):
                if i == range_index and each_res[i] > 0 and index_type != "Memory":
                    alarm_info += "<td bgcolor=green>{}</td>".format(each_res[i])
                elif i == range_index and each_res[i] > 0 and index_type == "Memory":
                    alarm_info += "<td bgcolor=red>{}</td>".format(each_res[i])
                elif i == range_index and each_res[i] < 0 and index_type != "Memory":
                    alarm_info += "<td bgcolor=red>{}</td>".format(each_res[i])
                elif i == range_index and each_res[i] < 0 and index_type == "Memory":
                    alarm_info += "<td bgcolor=green>{}</td>".format(each_res[i])
                else:
                    alarm_info += "<td>{}</td>".format(each_res[i])
            alarm_info += "</tr>\n"
        return alarm_info

    job_link = "" if single else "<td>job_link</td>"
    base_info = MAIL_MID_CONTENT.replace("JOB_LINK_HOLDER", job_link)
    res_info = ""
    if isinstance(results, list):
        alarm_info = get_alarm_info(results)
        res_info += base_info.replace("ALARM_INFO_HOLDER", alarm_info).replace("INDEX_HOLDER", "")
        return res_info if alarm_info else ""

    for index, values in results.items():
        alarm_info = get_alarm_info(values)
        if alarm_info:
            res_info += base_info.replace("ALARM_INFO_HOLDER",
                                          alarm_info).replace("INDEX_HOLDER", save.DICT_INDEX[index])
    return res_info


def construct_email_content(results, log_path, args, single=True):
    """construct email content.
    Args:
        results(dict): {1:[[], []], 2:[[], []]}
        log_path(str): email save path
        args(ArgParser): run args
        single(bool): single or multi_machine
    """
    print("html_results is {}".format(results))
    run_env = """
            <tr><td>paddle_branch</td><td>{}</td></tr>
            <tr><td>paddle_commit_id</td><td>{}</td></tr>
            <tr><td>benchmark_commit_id</td><td>{}</td></tr>
            <tr><td>device_type</td><td>{}</td></tr>
            <tr><td>implement_type</td><td>{}</td></tr>
             """.format(args.image_branch,
                        args.image_commit_id,
                        args.code_commit_id,
                        args.device_type,
                        args.implement_type, )

    if single and str(args.device_type).upper() in ("P40", "V100"):
        run_env += """
            <tr><td>cuda_version</td><td>{0}</td></tr>
            <tr><td>cudnn_version</td><td>{1}</td></tr>         
            <tr><td>docker_image</td><td>paddlepaddle/paddle:latest-gpu-cuda{0}-cudnn{1}</td></tr>     
            """.format(args.cuda_version, args.cudnn_version)
    elif single and str(args.device_type).upper() == "CPU":
        run_env += """
            <tr><td>docker_image</td><td>paddlepaddle/paddle:latest</td></tr>     
            """
    elif not single and str(args.device_type).upper() in ("P40", "V100"):
        run_env += """
            <tr><td>cuda_version</td><td>{0}</td></tr>
            <tr><td>cudnn_version</td><td>{1}</td></tr>         
            <tr><td>paddle_cloud_cluster</td><td>dltp-0-yq01-k8s-gpu-v100-8</td></tr>     
            """.format(args.cuda_version, args.cudnn_version)
    elif not single and str(args.device_type).upper() == "CPU":
        run_env += """   
            <tr><td>paddle_cloud_cluster</td><td>paddle_benchmark</td></tr>
            """
    alarm_info = construct_mid_content(results, single)
    if not alarm_info:
        return
    # Construct header of the message
    content = MAIL_HEAD_CONTENT.replace("RUN_ENV_HOLDER", run_env).strip()
    # Construct alarm content
    content += alarm_info
    # Construct the tail of the message
    content += MAIL_TAIL_CONTENT.replace("BENCHMARK_WEBSITE", os.getenv("BENCHMARK_WEBSITE", "")).strip()

    with open(os.path.join(log_path, "mail.html"), "w") as f_object:
        f_object.write(content)
