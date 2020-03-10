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
                             param += '--' + "variable" + '| '+ p_value["dtype"]+'| '
                         elif v_type == 0 and v_dtype ==1:
                             param += '--' + p_value["dtype"]+'| '
                         for param_key, param_value in p_value.items():
                             if param_key =="value":
                                 param += param_value + '| '
                             elif param_key != "type" and param_key != "dtype":
                                 param += param_key +':' + param_value + '| '
                         param = param[:-2]
                         param +='\n '
                    op_param = API(op, param)
                    op_whole.append(op_param)
    return model_name, op_whole

def connet_sql(model_name, op_whole, server):
    try:
        log_file = open(server)
    except Exception, e:
        print('File path not given or file not exists, please use "--=server_file" to set right server file')
        print('program exit\n')
        exit(0)

    host_name=''
    port_name=0
    user_name=''
    password=''
    db_name=''
    for line in log_file.readlines():
       line_k=line.split('=')[0]
       line_v=line.split('=')[1].replace("\n", "")
       if  line_k == 'host':
           host_name=line_v
       elif  line_k == 'port':
           port_name=int(line_v)
       elif  line_k == 'user':
           user_name=line_v
       elif  line_k == 'passwd':
           password=line_v
       elif  line_k == 'db':
           db_name=line_v

    print("save to database: case_from_model")
    db = pymysql.connect(host=host_name, port=port_name, user=user_name, 
                         passwd=password, db=db_name, charset='utf8')
    sql = "INSERT INTO case_from_model (case_name, op, param_info, model, update_time) \
           VALUES ('%s', '%s', '%s', '%s', '%d') ON DUPLICATE KEY UPDATE update_time=update_time"

    cursor = db.cursor()
    case_names=[]
    t = int(time.time())

    for i in op_whole:
        case_name = hashlib.sha224(str(i.op + i.param)).hexdigest()
        if case_name not in case_names:
            data=(case_name, i.op, i.param, model_name, t)
            case_names.append(case_name)
            cursor.execute(sql % data)
    
    sql_time="UPDATE case_from_model SET update_time=(%s)"
    cursor.execute(sql_time,(t))

    db.commit()


if __name__=='__main__':
    path, server = arg_parse()
    model_name, op = process_api_log(path)
    print(model_name)
    connet_sql(model_name, op, server)
    print("process finished!!")


