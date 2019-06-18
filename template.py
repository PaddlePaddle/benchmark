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
"""
import os

def construct_email_content(results, log_path, args):
    """the list not satify condition"""
    # if not results:
    #     return

    content = """
From:paddle_benchmark@baidu.com
To:xxx@yy.com
Subject:test_benchmark
Subject:benchmark运行结果报警，请检查
content-type:text/html
<html>
    <body>
        <h3 align=center>benchmark alarm email</h3>
        <HR align=center width="80%" SIZE=1>         
        <table border="1" align=center>
        <caption bgcolor="#989898">环境配置</caption>
            <tr><td>paddle_commit_id</td><td>{}</td></tr>
            <tr><td>benchmark_commit_id</td><td>{}</td></tr>
            <tr><td>cuda_version</td><td>{}</td></tr>
            <tr><td>cudnn_version</td><td>{}</td></tr>
            <tr><td>gpu_type</td><td>{}</td></tr>
            <tr><td>implement_type</td><td>{}</td></tr>
            <tr><td>docker_image</td><td>paddlepaddle/paddle:latest-gpu-cuda{}-cudnn{}</td></tr>
        </table>
        <HR align=center width="80%" SIZE=1>
        <table border="1" align=center>
        <caption>报警结果</caption>
            <tr bgcolor="#989898" ><td>模型</td><td>运行环境</td><td>指标</td><td>标准值</td><td>当前值</td><td>波动范围</td></tr>
place_holder
        </table>
        <HR align=center width="80%" SIZE=1>
        <h4 align=center>历史详细数据 http://xxxxxx:yyy/</h4> 
    </body>
</html>
""".format(args.image_commit_id,
           args.code_commit_id,
           args.cuda_version,
           args.cudnn_version,
           args.gpu_type,
           args.implement_type,
           args.cuda_version,
           args.cudnn_version)

    place_holder = ""
    for result in results:
        if isinstance(result, list):
            place_holder += "            <tr>"
            for i in range(len(result)):
                if i == len(result)-1 and result[i] > 0:
                    place_holder += "<td bgcolor=green>{}</td>".format(result[i])
                elif i == len(result)-1 and result[i] < 0:
                    place_holder += "<td bgcolor=red>{}</td>".format(result[i])
                else:
                    place_holder += "<td>{}</td>".format(result[i])

            place_holder += "</tr>\n"

    content = content.replace("place_holder", place_holder).strip()

    with open(os.path.join(log_path, "mail.html"), "w") as f_object:
        f_object.write(content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args =parser.parse_args()
    args.cuda_version = "9.0"
    args.cudnn_version = 7
    args.image_commit_id = ""
    args.code_commit_id = ""
    results = [["CycleGan", "ONE_GPU", "speed", 100, 95, -0.05],
          ["rcnn", "ONE_GPU", "speed", 100, 95, 0.05],
          ["test", "ONE_GPU", "speed", 100, 95, -0.05],]
    construct_email_content(results, "./", args)
