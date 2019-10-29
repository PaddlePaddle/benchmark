#!/usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================

"""
@Desc: template module
@File: template.py
@Author: liangjinhua
@Date: 2019/5/30 19:26
"""
import os

MAIL_HEAD_CONTENT = """
From:paddle_benchmark@baidu.com
To:liangjinhua01@baidu.com
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

MAIL_TAIL_CONTENT = """       
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption>报警结果</caption>
            <tr bgcolor="#989898" ><td>模型</td><td>运行环境</td><td>指标</td><td>标准值</td><td>当前值</td><td>波动范围</td></tr>
ALARM_INFO_HOLDER
        </table>
        <HR align=center width="80%" SIZE=1>
        <h4 align=center>历史详细数据 http://yq01-page-powerbang-table1077.yq01.baidu.com:8988/</h4> 
    </body>
</html>
"""


def construct_email_content(results, log_path, args):
    """the list not satify condition"""
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
                        args.implement_type,)

    if str(args.device_type).upper() in ("P40", "V100") and args.job_type == 2:
        run_env += """          
            <tr><td>cuda_version</td><td>{0}</td></tr>
            <tr><td>cudnn_version</td><td>{1}</td></tr>         
            <tr><td>docker_image</td><td>paddlepaddle/paddle:latest-gpu-cuda{0}-cudnn{1}</td></tr>     
            """.format(args.cuda_version, args.cudnn_version)
    elif str(args.device_type).upper() == "CPU" and args.job_type == 2:
        run_env += """                
            <tr><td>docker_image</td><td>paddlepaddle/paddle:latest</td></tr>     
            """
    elif str(args.device_type).upper() in ("P40", "V100") and args.job_type == 5:
        run_env += """          
            <tr><td>cuda_version</td><td>{0}</td></tr>
            <tr><td>cudnn_version</td><td>{1}</td></tr>         
            <tr><td>paddle_cloud_cluster</td><td>dltp-0-yq01-k8s-gpu-v100-8</td></tr>     
            """.format(args.cuda_version, args.cudnn_version)
    elif str(args.device_type).upper() == "CPU" and args.job_type == 5:
        run_env += """                
            <tr><td>paddle_cloud_cluster</td><td>paddle_benchmark</td></tr>
            """

    place_holder = ""
    for result in results:
        if isinstance(result, list):
            place_holder += "            <tr>"
            index_type = ""
            for i in range(len(result)):
                if i == len(result)-1 and result[i] > 0 and index_type != "mem":
                    place_holder += "<td bgcolor=green>{}</td>".format(result[i])
                elif i == len(result)-1 and result[i] > 0 and index_type == "mem":
                    place_holder += "<td bgcolor=red>{}</td>".format(result[i])
                elif i == len(result)-1 and result[i] < 0 and index_type != "mem":
                    place_holder += "<td bgcolor=red>{}</td>".format(result[i])
                elif i == len(result)-1 and result[i] < 0 and index_type == "mem":
                    place_holder += "<td bgcolor=green>{}</td>".format(result[i])
                else:
                    place_holder += "<td>{}</td>".format(result[i])
                    if str(result[i]) in ("mem", 'speed', 'maxbs'):
                        index_type = str(result[i])

            place_holder += "</tr>\n"
    content = MAIL_HEAD_CONTENT.replace("RUN_ENV_HOLDER", run_env).strip()
    content += MAIL_TAIL_CONTENT.replace("ALARM_INFO_HOLDER", place_holder).strip()

    with open(os.path.join(log_path, "mail.html"), "w") as f_object:
        f_object.write(content)

