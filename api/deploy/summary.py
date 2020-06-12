#!/bin/python
"""
summary script
"""
from __future__ import print_function
import os
import json
import xlsxwriter as xlw
import time
import sys

res = {}
path = sys.argv[1]

def dump_excel(data):
    """
    dump data to a excel
    """
    wb = xlw.Workbook('Operators.xlsx')
    ws = wb.add_worksheet('OP')

    align = wb.add_format({'align': 'right'})
    bold = wb.add_format({'bold': 15, 'color': 'black'})
    wrong_format = wb.add_format({'bold': 8, 'color': 'red', 'align': 'right'})

    ws.set_row(0, 15, bold)
    ws.set_column(0, 1, 15)
    ws.set_column(1, 2, 28)
    ws.set_column(2, 3, 28)
    ws.set_column(3, 4, 22)
    ws.set_column(4, 5, 22)
    ws.set_column(5, 6, 22)
    ws.set_column(6, 7, 22)
    ws.set_column(7, 8, 22)
    ws.set_column(8, 9, 22)
    ws.set_column(9, 10, 22)
    ws.set_column(10, 11, 22)

    ws.set_row(0, 15, align)
    ws.set_column(0, 1, 15, align)
    ws.set_column(1, 2, 28, align)
    ws.set_column(2, 3, 28, align)
    ws.set_column(3, 4, 22, align)
    ws.set_column(4, 5, 22, align)
    ws.set_column(5, 6, 22, align)
    ws.set_column(6, 7, 22, align)
    ws.set_column(7, 8, 22, align)
    ws.set_column(8, 9, 22, align)
    ws.set_column(9, 10, 22, align)
    ws.set_column(10, 11, 22, align)

    row = 0
    column = 0
    ws.write(row, column, 'name')
    ws.write(row, column + 1, 'paddle_cpu_accuracy')
    ws.write(row, column + 2, 'paddle_gpu_accuracy')
    ws.write(row, column + 3, 'paddle_cpu_perf(ms)')
    ws.write(row, column + 4, 'tf_cpu_perf(ms)')
    ws.write(row, column + 5, 'paddle_gpu_perf(ms)')
    ws.write(row, column + 6, 'tf_gpu_perf(ms)')
    ws.write(row, column + 7, 'paddle_cpu_perf_backwards(ms)')
    ws.write(row, column + 8, 'tf_cpu_perf_backwards(ms)')
    ws.write(row, column + 9, 'paddle_gpu_perf_backwards(ms)')
    ws.write(row, column + 10, 'tf_gpu_perf_backwards(ms)')

    row = 1
    column = 0
    for i in range(len(data)):
        for key, value in data[i].items():
            if key == 'name':
                val = value.split("-")[0]
                ws.write(row, column, val)
            elif key == 'paddle_cpu_accuracy':
                if not value:
                    ws.write(row, column + 1, value, wrong_format)
                else:
                    ws.write(row, column + 1, value)
            elif key == 'paddle_gpu_accuracy':
                if not value:
                    ws.write(row, column + 2, value, wrong_format)
                else:
                    ws.write(row, column + 2, value)
            elif key == 'paddle_cpu_perf':
                ws.write_string(row, column + 3, value)
            elif key == 'tf_cpu_perf':
                ws.write_string(row, column + 4, value)
            elif key == 'paddle_gpu_perf':
                ws.write_string(row, column + 5, value)
            elif key == 'tf_gpu_perf':
                ws.write_string(row, column + 6, value)
            elif key == 'paddle_cpu_perf_backwards':
                ws.write_string(row, column + 7, value)
            elif key == 'tf_cpu_perf_backwards':
                ws.write_string(row, column + 8, value)
            elif key == 'paddle_gpu_perf_backwards':
                ws.write_string(row, column + 9, value)
            elif key == 'tf_gpu_perf_backwards':
                ws.write_string(row, column + 10, value)
            else:
                pass
        row += 1

    wb.close()

def get_job_res(inputfile, statistic_file):
    """
    implements within avoiding  too large file
    inputfile -- directory path
    statistic_file -- statistic by one dimension
    """
    filename = os.path.splitext(statistic_file)[0]
    case_name = filename.split("-")[0]
    statistic_beg_idx = filename.find("-")
    statistic_type = filename[statistic_beg_idx + 1 :]
    filesize = os.path.getsize(inputfile)
    blocksize = 1024
    data_file = open(inputfile, 'r')
    if filesize > blocksize:
        maxseekpoint = filesize // blocksize
        data_file.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        data_file.seek(0, 0)
    lines = data_file.readlines()
    d = {}
    param = ""
    try:
        last_line = lines[-2].strip("\n")
        d = json.loads(last_line)
        param = d['parameters']
    except Exception:
        if case_name not in res:
            res[case_name] = {}
            res[case_name][statistic_type] = "--"
        else:
            res[case_name][statistic_type] = "--"
    if lines and "_speed_" in inputfile:
        try:
            dic = json.loads(last_line)
            perf = dic['speed']['total']
            if case_name not in res:
                res[case_name] = {}
            res[case_name][statistic_type] = str(perf)
        except Exception:
            if case_name not in res:
                res[case_name] = {}
                res[case_name][statistic_type] = "--"
            else:
                res[case_name][statistic_type] = "--"
    if lines and "_accuracy_" in inputfile:
        try:
            dic = json.loads(last_line)
            consitent_status = dic['consistent']
            if case_name not in res:
                 res[case_name] = {}
            res[case_name][statistic_type] = consitent_status
        except Exception:
            if case_name not in res:
                 res[case_name] = {}
                 res[case_name][statistic_type] = "--"
            else:
                 res[case_name][statistic_type] = "--"
    if lines and "paddle_gpu_speed_backward" in inputfile:
        try:
            dic = json.loads(last_line)
            gpu_time = dic['speed']['gpu_time']
            if case_name not in res:
                res[case_name] = {}
            res[case_name]['gpu_time_backward'] = str(gpu_time)
        except Exception:
            if case_name not in res:
                res[case_name] = {}
                res[case_name]['gpu_time_backward'] = "--"
            else:
                res[case_name]['gpu_time_backward'] = "--"
    if lines and "tensorflow_gpu_speed_backward" in inputfile:
        try:
            dic = json.loads(last_line)
            gpu_time = dic['speed']['gpu_time']
            if case_name not in res:
                res[case_name] = {}
            res[case_name]['tf_gpu_time_backward'] = str(gpu_time)
        except Exception:
            if case_name not in res:
                res[case_name] = {}
                res[case_name]['tf_gpu_time_backward'] = "--"
            else:
                res[case_name]['tf_gpu_time_backward'] = "--"
    if lines and "paddle_gpu_speed_forward" in inputfile:
        try:
            dic = json.loads(last_line)
            gpu_time = dic['speed']['gpu_time']
            if case_name not in res:
                res[case_name] = {}
            res[case_name]['gpu_time'] = str(gpu_time)
        except Exception:
            if case_name not in res:
                res[case_name] = {}
                res[case_name]['gpu_time'] = "--"
            else:
                res[case_name]['gpu_time'] = "--"
    if lines and "tensorflow_gpu_speed_forward" in inputfile:
        try:
            dic = json.loads(last_line)
            gpu_time = dic['speed']['gpu_time']
            if case_name not in res:
                res[case_name] = {}
            res[case_name]['tf_gpu_time'] = str(gpu_time)
        except Exception:
            if case_name not in res:
                res[case_name] = {}
                res[case_name]['tf_gpu_time'] = "--"
            else:
                res[case_name]['tf_gpu_time'] = "--"
    if param != "":
        res[case_name]['parameters'] = param.strip("\n")
    data_file.close()
    return res


def dump_mysql(data):
    """
    dump data to mysql database
    """
    timestamp = time.time()
    for i in range(len(data)):
        dic = data[i]
        case_name = dic['name']
        paddle_cpu_accuracy = "--"
        paddle_cpu_accuracy_backwards = "--"
        paddle_gpu_accuracy = "--"
        paddle_gpu_accuracy_backwards = "--"
        paddle_cpu_perf = "--"
        tf_cpu_perf = "--"
        paddle_gpu_perf = "--"
        tf_gpu_perf = "--"
        paddle_cpu_perf_backwards = "--"
        tf_cpu_perf_backwards = "--"
        paddle_gpu_perf_backwards = "--"
        tf_gpu_perf_backwards = "--"
        parameters = "--"
        gpu_time = "--"
        gpu_time_backward = "--"
        tf_gpu_time = "--"
        tf_gpu_time_backward = "--"

        for k, v in dic.items():
            if k == "paddle_cpu_accuracy_forward":
                paddle_cpu_accuracy = v
            elif k == "paddle_cpu_accuracy_backward":
                paddle_cpu_accuracy_backwards = v
            elif k == "paddle_gpu_accuracy_forward":
                paddle_gpu_accuracy = v
            elif k == "paddle_gpu_accuracy_backward":
                paddle_gpu_accuracy_backwards = v
            elif k == "paddle_cpu_speed_forward":
                paddle_cpu_perf = v
            elif k == "tensorflow_cpu_speed_forward":
                tf_cpu_perf = v
            elif k == "paddle_gpu_speed_forward":
                paddle_gpu_perf = v
            elif k == "tensorflow_gpu_speed_forward":
                tf_gpu_perf = v
            elif k == "paddle_cpu_speed_backward":
                paddle_cpu_perf_backwards = v
            elif k == "tensorflow_cpu_speed_backward":
                tf_cpu_perf_backwards = v
            elif k == "paddle_gpu_speed_backward":
                paddle_gpu_perf_backwards = v
            elif k == "tensorflow_gpu_speed_backward":
                tf_gpu_perf_backwards = v
            elif k == "parameters":
                parameters = v
            elif k == "gpu_time_backward":
                gpu_time_backward = v
            elif k == "gpu_time":
                gpu_time = v
            elif k == "tf_gpu_time_backward":
                tf_gpu_time_backward = v
            elif k == "tf_gpu_time":
                tf_gpu_time = v
            else:
                pass

        cmd = 'nvidia-docker exec mysql ./mysql -e "insert into paddle.op_record2 ' \
              'values(\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', {}, \'{}\', \'{}\', \'{}\', \'{}\')' \
              'on duplicate key update case_name=\'{}\', paddle_cpu_accuracy=\'{}\', paddle_cpu_accuracy_backwards=\'{}\', paddle_gpu_accuracy=\'{}\', paddle_gpu_accuracy_backwards=\'{}\', paddle_cpu_perf=\'{}\',' \
              'tf_cpu_perf=\'{}\', paddle_gpu_perf=\'{}\', tf_gpu_perf=\'{}\', paddle_cpu_perf_backwards=\'{}\', tf_cpu_perf_backwards=\'{}\',' \
              'paddle_gpu_perf_backwards=\'{}\', tf_gpu_perf_backwards=\'{}\', log_url= \'{}\', config=\'{}\', timestamp={}, gpu_time=\'{}\', gpu_time_backward=\'{}\', tf_gpu_time=\'{}\', tf_gpu_time_backward=\'{}\';"'\
            .format(case_name, paddle_cpu_accuracy, paddle_cpu_accuracy_backwards, paddle_gpu_accuracy, paddle_gpu_accuracy_backwards,
                    paddle_cpu_perf, tf_cpu_perf, paddle_gpu_perf, tf_gpu_perf, paddle_cpu_perf_backwards,
                    tf_cpu_perf_backwards, paddle_gpu_perf_backwards, tf_gpu_perf_backwards, "--", parameters, timestamp, gpu_time, gpu_time_backward, tf_gpu_time, tf_gpu_time_backward,
                    case_name, paddle_cpu_accuracy, paddle_cpu_accuracy_backwards, paddle_gpu_accuracy, paddle_gpu_accuracy_backwards,
                    paddle_cpu_perf, tf_cpu_perf, paddle_gpu_perf, tf_gpu_perf, paddle_cpu_perf_backwards,
                    tf_cpu_perf_backwards, paddle_gpu_perf_backwards, tf_gpu_perf_backwards, "--", parameters, timestamp, gpu_time, gpu_time_backward, tf_gpu_time, tf_gpu_time_backward
                    )
        os.system(cmd)


dirs = os.listdir(path)
dirs.remove('api_info.txt')


for d in dirs:
    res = get_job_res(os.path.join(path, d), d)

data = []
excel_dic = {}

for k, v in res.items():
    excel_dic = v.copy()
    excel_dic['name'] = k
    data.append(excel_dic)
print(data)

try:
    dump_excel(data)
except Exception as e:
    print(e)
    print("write excel failed, please check the reason!")
try:
    dump_mysql(data)
except Exception as e:
    print(e)
    print("dump data into mysql failed, please check reason!")

