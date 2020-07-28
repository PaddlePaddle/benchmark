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

MAIL_HEAD_CONTENT = """From:paddle_benchmark@baidu.com
To:test@benchmark.com
Subject:benchmark运行结果报警，请检查
content-type:text/html
<html>
    <body>
        <h3 align=center>TITLE_HOLDER alarm email</h3>
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
            <tr bgcolor="#989898" >
TABLE_HEADER_HOLDER
            </tr>
ALARM_INFO_HOLDER
        </table>
"""

MAIL_TAIL_CONTENT = """
        <HR align=center width="80%" SIZE=1>
        <h4 align=center>历史详细数据 BENCHMARK_WEBSITE</h4> 
    </body>
</html>
"""


class EmailTemplate(object):
    """construct email for benchmark result.
    """
    def __init__(self, title, env, results, log_path):
        """
        Args:
            title(str): benchmark | op_benchmark | benchmark_distribute
            env(dict): running environment.
            results(dict):
                {"Speed": {"header": [table_header0, table_header1, table_header2,]
                           "data": [[{'value':, 'color':, }, {'value':, 'color':, }, {'value':, 'color':, }]
                           ...]}
                "Mem": {"header": [table_header0, table_header1, table_header2,]
                        "data": [[{'value':, 'color':, }, {'value':, 'color':, }, {'value':, 'color':, }]
                        ...]}
                ...}
            log_path(str): mail path
        """
        self.title = title
        self.env_content = ""
        self.alarm_info = ""
        self.log_path = log_path
        self.__construct_mail_env(env)
        self.__construct_alarm_info(results)

    def __construct_mail_env(self, env):
        """
        construct email env content.
        """

        if isinstance(env, dict):
            for k, v in env.items():
                self.env_content += """
                    <tr><td>{}</td><td>{}</td></tr>
                    """.format(k, v)
        return self.env_content

    def __construct_alarm_info(self, results):
        """
        construct email env content.
        """
        print("html_results is: {}".format(results))
        for report_index, alarm in results.items():
            table_header_info = ""
            table_alarm_info = ""
            if not alarm.get("data"):
                continue
            for header_td in alarm["header"]:
                table_header_info += """
                    <td>{}</td>
                    """.format(header_td)
            for single_info in alarm["data"]:
                if not single_info:
                    continue
                table_alarm_info += "\t\t\t<tr>"
                for info_td in single_info:
                    table_alarm_info += """
                    <td bgcolor={}>{}</td>
                    """.format(info_td.get('color', 'white'), info_td.get('value'))
                table_alarm_info += "</tr>\n"
            if table_alarm_info:
                self.alarm_info += MAIL_MID_CONTENT.replace('TABLE_HEADER_HOLDER', table_header_info).replace(
                    'ALARM_INFO_HOLDER', table_alarm_info).replace('INDEX_HOLDER', report_index)

    def construct_email_content(self):
        """
        construct email full content.
        """
        # Construct header of the message
        content = MAIL_HEAD_CONTENT.replace("TITLE_HOLDER", self.title).replace('RUN_ENV_HOLDER',
                                                                                self.env_content)
        if not self.alarm_info:
            return
        # Construct alarm content
        content += self.alarm_info
        # Construct the tail of the message
        content += MAIL_TAIL_CONTENT.replace("BENCHMARK_WEBSITE", os.getenv("BENCHMARK_WEBSITE", "")).strip()

        with open(os.path.join(self.log_path, "mail.html"), "w") as f_object:
            f_object.write(content)
