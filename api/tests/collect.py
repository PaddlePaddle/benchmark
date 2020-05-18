#!/bin/python
"""
gather script

"""
from __future__ import print_function
import os
import json
import xlsxwriter as xlw

res = {}
path = "./result/"


def get_job_res(inputfile, statistic_file):
    """
    implements within avoiding  too large file
    inputfile -- directory path
    statistic_file -- statistic by one dimension

    """
    op_name = statistic_file.split("_")[0]
    statistic_file_length = len(statistic_file)
    statistic_type = statistic_file[statistic_file.index(op_name) + len(op_name) + 1: statistic_file_length - 4]
    filesize = os.path.getsize(inputfile)
    blocksize = 1024
    data_file = open(inputfile, 'r')
    if filesize > blocksize:
        maxseekpoint = filesize // blocksize
        data_file.seek((maxseekpoint - 1) * blocksize)
    elif filesize:
        data_file.seek(0, 0)
    lines = data_file.readlines()
    if lines and "perf" in inputfile:
        last_line = lines[-2].strip("\n")
        s = last_line
        try:
            begin = s.index('total')
            s1 = s[begin:]
            end = s1.index(',')
            s2 = s1[:end]
            need_str = s2.split(" ")
            perf = need_str[1]
            if op_name not in res:
                res[op_name] = {}
            res[op_name][statistic_type] = perf
        except Exception:
            if op_name not in res:
                res[op_name] = {}
                res[op_name][statistic_type] = "--"
            else:
                res[op_name][statistic_type] = "--"
    if lines and "accuracy" in inputfile:
        last_line = lines[-1].strip()
        try:
            dic = json.loads(last_line)
            consitent_status = dic['consistent']
            if op_name not in res:
                 res[op_name] = {}
            res[op_name][statistic_type] = consitent_status
        except Exception:
            if op_name not in res:
                 res[op_name] = {}
                 res[op_name][statistic_type] = "--"
            else:
                 res[op_name][statistic_type] = "--"
    data_file.close()
    return res


dirs = os.listdir(path)
# print(dirs)

for d in dirs:
    res = get_job_res(os.path.join(path, d), d)
# print(res)

data = []
excel_dic = {}

for k, v in res.items():
    excel_dic = v.copy()
    excel_dic['name'] = k
    data.append(excel_dic)
print(data)

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

ws.set_row(0, 15, align)
ws.set_column(0, 1, 15, align)
ws.set_column(1, 2, 28, align)
ws.set_column(2, 3, 28, align)
ws.set_column(3, 4, 22, align)
ws.set_column(4, 5, 22, align)
ws.set_column(5, 6, 22, align)
ws.set_column(6, 7, 22, align)

row = 0
column = 0
ws.write(row, column, 'name')
ws.write(row, column + 1, 'paddle_cpu_accuracy')
ws.write(row, column + 2, 'paddle_gpu_accuracy')
ws.write(row, column + 3, 'paddle_cpu_perf(ms)')
ws.write(row, column + 4, 'tf_cpu_perf(ms)')
ws.write(row, column + 5, 'paddle_gpu_perf(ms)')
ws.write(row, column + 6, 'tf_gpu_perf(ms)')

row = 1
column = 0
for i in range(len(data)):
    for key, value in data[i].items():
        if key == 'name':
            val = value.split("-")[0]
            ws.write(row, column, val)
        if key == 'paddle_cpu_accuracy':
            if not value:
                ws.write(row, column + 1, value, wrong_format)
            else:
                ws.write(row, column + 1, value)
        if key == 'paddle_gpu_accuracy':
            if not value:
                ws.write(row, column + 2, value, wrong_format)
            else:
                ws.write(row, column + 2, value)
        if key == 'paddle_cpu_perf':
            ws.write_string(row, column + 3, value)
        if key == 'tf_cpu_perf':
            ws.write_string(row, column + 4, value)
        if key == 'paddle_gpu_perf':
            ws.write_string(row, column + 5, value)
        if key == 'tf_gpu_perf':
            ws.write_string(row, column + 6, value)
    row += 1

wb.close()
