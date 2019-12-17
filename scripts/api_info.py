import os
import string
import pymysql
import hashlib
import re
import time
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument(
        '--log_file',
        type=str,
        default='none',
        help='The path of the log file')

    args = parser.parse_args()
    path = args.log_file
    return path 

def process_api_log(path):
    ops_para = list(dict())
    op_parameter=dict()

    try:
        log_file = open(path)
    except Exception, e:
        print('File path not given or file not exists, please use "--log_file=path" to set right log file')
        print('program exit\n')
        exit(0)

    name, model_name = os.path.splitext(path)

# This is for processing model log
#path=os.environ.get('API_PATH')
#line = file.readline()
#process log header
#while re.match('\-+\W\Z', line) is None:
#    line = file.readline()
#    key = line.split(':')[0].strip()
#    if key  == "model":
#        model_name=line.split(':')[1].strip()

    model_name = model_name.split('.')[-1]

# process api
    for line in log_file.readlines():
        vars_ = []
    # OP name
        if "|" not in line:
            line = line.split(':')[-1]
            op_name = filter(lambda x: x in string.ascii_letters + string.digits + '_', line)
            if op_parameter.values(): 
                ops_para.append(op_parameter)
                op_parameter = dict()
            op_parameter[op_name] = []
        #parameter
        elif '|' in line:
            line_sps = line.split('|')
            name_ = line_sps[0].split()[0].replace("\n", "")
            type_ = line_sps[0].split()[1].replace("\n", "")
            vars_.append(name_)
            vars_.append(type_)
            values = ""
            if vars_[1] == "Variable":
                vars_.append(line_sps[1].replace("\n", ""))
                vars_.append(line_sps[2].replace("\n", ""))
            else:
                vars_.append(line_sps[1].replace("\n", ""))
            para = vars_[0] + '--' + vars_[1]
            lvar = vars_[2:]
            for var in lvar:
                para = para + '|' + var
            op_parameter[op_name].append(para)
    if op_parameter.values():
        ops_para.append(op_parameter)        

    print("save to database: case_from_model")
    db = pymysql.connect(host='gzbh-qianmo-com-162-69-149.gzbh.baidu.com',
                         port=3306, user='root', passwd='', db='paddle', charset='utf8') 
    cursor = db.cursor()
    table_name='case_from_model'

    #cursor.execute("TRUNCATE TABLE %s" % table_name)
    cursor.execute("select * from %s" % table_name)
    res = cursor.fetchall()
    db.commit()
    #print(res)
    sql = "INSERT INTO case_from_model (case_name, op, param_info, model, update_time) \
           VALUES ('%s', '%s', '%s', '%s', '%d') ON DUPLICATE KEY UPDATE update_time=update_time"
    sql_main = "INSERT INTO case_from_model (case_name) VALUES ('%s')"

    case_names=[]
    op_counts=dict()
    for v in ops_para:
        for key, pa in v.items():
            op_counts[key] = 0

    for v in ops_para:
        # key save op name, paras save all parameter
        for key, paras in v.items():
            op_counts[key] +=1;
            case_name = hashlib.sha224(str(paras)).hexdigest()
            if case_name not in case_names and op_counts[key] < 5:
                para_info=""
                for p in paras:
                    para_info  += p
                    para_info += '\n '
                t = int(time.time())
                data=(case_name, key, para_info, model_name, t)
                cursor = db.cursor()
                cursor.execute(sql % data)
                case_names.append(case_name)

    db.commit()

if __name__=='__main__':
    path = arg_parse()
    process_api_log(path)
    print("process finished!!!")
