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

import json
import time
import pymysql
import hashlib
import argparse
import sys
from api_param import APIParam


class APIDatabase(object):
    def __init__(self, host, port, user, passwd, db):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db

    def _connect(self):
        self.cdb = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            passwd=self.passwd,
            db=self.db,
            charset='utf8')

    def update(self, table, api_list, model):
        cursor = self.cdb.cursor()
        case_names = []
        sql = "INSERT INTO %s" % table + " (case_name, op, param_info, model, update_time) \
           VALUES ('%s', '%s', '%s', '%s', '%d') ON DUPLICATE KEY UPDATE update_time=update_time"

        t = int(time.time())
        for i in api_list:
            params_str = i._convert_params_to_str()
            case_name = hashlib.sha224(str(i.name + params_str)).hexdigest()
            if case_name not in case_names:
                print('write data to database')
                data = (case_name, i.name, params_str, model, t)
                case_names.append(case_name)
                cursor.execute(sql % data)
        sql_time = "UPDATE %s" % table + " SET update_time=(%s)"
        cursor.execute(sql_time, (t))
        self.cdb.commit()


def arg_parse():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument(
        '--log_file',
        type=str,
        default='none',
        help='The path of the log file')

    parser.add_argument(
        '--server_file',
        type=str,
        default='none',
        help='The configuration of server')

    args = parser.parse_args()
    path = args.log_file
    server = args.server_file
    return path, server


def server_param(server):
    try:
        log_file = open(server)
    except Exception, e:
        print(
            'File path not given or file not exists, please use "--=server_file" to set right server file'
        )
        print('program exit\n')
        exit(0)
    host_name = ''
    port_name = 0
    user_name = ''
    password = ''
    db_name = ''
    for line in log_file.readlines():
        line_k = line.split('=')[0]
        line_v = line.split('=')[1].replace("\n", "")
        if line_k == 'host':
            host_name = line_v
        elif line_k == 'port':
            port_name = int(line_v)
        elif line_k == 'user':
            user_name = line_v
        elif line_k == 'passwd':
            password = line_v
        elif line_k == 'db':
            db_name = line_v

    server_info = APIDatabase(host_name, port_name, user_name, password,
                              db_name)
    return server_info


def save_api_params(filename, server_info):
    data_size = 0
    server_info._connect()
    api_list = []
    table = "case_from_model"
    with open(filename, 'r') as f:
        data = json.load(f)
        data_size = len(data)
        model = data[data_size - 1]["model"]
    for i in range(0, data_size - 1):
        api_param = APIParam('', '')
        api_param.init_from_json(filename, i)
        api_list.append(api_param)
    server_info.update(table, api_list, model)


if __name__ == '__main__':
    filename, server = arg_parse()
    server_info = server_param(server)
    save_api_params(filename, server_info)
