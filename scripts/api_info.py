import os
import json
import string
import pymysql
import hashlib
import re
import time
import argparse

class API:
    def __init__(self, op, param):
        self.op = op
        self.param = param
   
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

def process_api_log(path):
    op=""
    op_whole=[]
    model_name=""
    with open(path, 'r') as f:
        data = json.load(f)
        for i in range(0,len(data)):
            print(i)
            for key, value in data[i].items():
                param=""
                if key == "model":
                    model_name=value
                elif key == "param_info":
                    op = data[i]["op"]
                    for p_key, p_value in value.items():
                         param += p_key
                         v_type = 0
                         v_dtype = 0
                         for v_k in p_value.keys():
                             if v_k == "type":
                                 v_type =1
                             if v_k == "dtype":
                                 v_dtype = 1
                         if v_type == 1 and v_dtype ==1:
                             param += '--' + "variable" + '|'+ p_value["dtype"]+'|'
                         elif v_type == 0 and v_dtype ==1:
                             param += '--' + p_value["dtype"]+'|'
                         for param_key, param_value in p_value.items():
                             if param_key =="value":
                                 param += param_value + '|'
                             elif param_key != "type" and param_key != "dtype":
                                 param += param_key +' : ' + param_value + '|'
                    param = param[:-1]
                    param += '\n'
                print(param)
                op_param = API(op, param)
                op_whole.append(op_param)
                   # print(op_param.op)
                print(op_param.param)
    for i in op_whole:
        print(i.op)
        print(i.param)
    return op_whole

def connet_sql(op_whole, server):
    print("save to database: case_from_model")
    sql = "INSERT INTO case_from_model (case_name, op, param_info, model, update_time) \
           VALUES ('%s', '%s', '%s', '%s', '%d') ON DUPLICATE KEY UPDATE update_time=update_time"

    cursor = db.cursor()
    case_names=[]

    for i in op_whole:
        #print(op_name[i])
        #print(op_parameter[i])
        case_name = hashlib.sha224(str(i.op + i.param)).hexdigest()
        if case_name not in case_names:
            t = int(time.time())
            data=(case_name, i.op, i.param, model_name, t)
            case_names.append(case_name)
            cursor.execute(sql % data)

    db.commit()


if __name__=='__main__':
    path, server = arg_parse()
    op = process_api_log(path)
    connet_sql(op, server)
    print("process finished!!")


