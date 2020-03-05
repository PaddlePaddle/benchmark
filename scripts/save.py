#!/usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================

"""
@Desc: db module
@File: db.py
@Author: liangjinhua
@Date: 2019/5/5 19:30
"""
import argparse
import os
import sys
import time
import uuid
import subprocess
import numpy as np
import template
import socket
import json
import commands

base_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print sys.path
import models.benchmark_server.helper as helper
from benchmark_server import benchmark_models as bm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--log_path",
    type=str,
    default='/home/crim/benchmark/logs',
    help="The cases files. (default: %(default)d)")

parser.add_argument(
    "--code_commit_id",
    type=str,
    default='',
    help="The benchmark repo commit id")

parser.add_argument(
    "--image_commit_id",
    type=str,
    default='',
    help="The benchmark repo commit id")

parser.add_argument(
    "--image_branch",
    type=str,
    default='develop',
    help="The benchmark repo branch")

parser.add_argument(
    "--cuda_version",
    type=str,
    default='9.0',
    help="The benchmark run on cuda version")

parser.add_argument(
    "--cudnn_version",
    type=str,
    default='7',
    help="The benchmark run on cudnn version")

parser.add_argument(
    "--paddle_version",
    type=str,
    default='test',
    help="The benchmark run on paddle whl version")

parser.add_argument(
    "--job_type",
    type=int,
    default=2,
    help="The benchmark job_type")

parser.add_argument(
    "--device_type",
    type=str,
    default='v100',
    help="The benchmark run on v100 or p40")

parser.add_argument(
    "--implement_type",
    type=str,
    default="staticgraph",
    help="The benchmark model implement method")

DICT_RUN_MACHINE_TYPE = {'1': 'ONE_GPU', '4': 'FOUR_GPU', 
                            '8': 'MULTI_GPU', '8mp': 'MULTI_GPU_MULTI_PROCESS'}
DICT_INDEX = {1: "Speed", 2: "Memory", 3: "Profiler_info", 6: "Max_bs"}
# todo config the log_server port
LOG_SERVER = "http://" + socket.gethostname()+ ":8777/"


def load_folder_files(folder_path, recursive=True):
    """
    :param folder_path: specified folder path to load
    :param recursive: if True, will load files recursively
    :return:
    """
    if isinstance(folder_path, (list, set)):
        files = []
        for path in set(folder_path):
            files.extend(load_folder_files(path, recursive))

        return files

    if not os.path.exists(folder_path):
        return []

    file_list = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        filenames_list = []

        for filename in filenames:
            filenames_list.append(filename)

        for filename in filenames_list:
            file_path = os.path.join(dirpath, filename)
            file_list.append(file_path)

        if not recursive:
            break

    return file_list


def get_image_id():
    """
    :return:
    """
    cur_time = time.time()
    ct = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(cur_time))
    if args.image_branch == "develop":
        image_branch = "develop"
    elif args.image_branch.isdigit():
        image_branch = "pull_requests"
    else:
        image_branch = "release"

    pi = bm.Image()
    pi.frame_id = 0
    pi.version = args.paddle_version
    pi.cuda_version = args.cuda_version
    pi.cudnn_version = args.cudnn_version
    pi.image_commit_id = args.image_commit_id
    pi.image_branch = image_branch
    pi.image_type = args.job_type
    pi.create_time = ct
    pi.save()
    return pi.image_id


def send_email(title, mailto, cc, content):
    """send email"""
    try:
        p = subprocess.Popen(['mail', '-s', title, '-c', cc, mailto],
                             stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate(input=content.encode('utf8'))
        print out, err
    except Exception as e:
        print e


def check_results(job_info, run_machine_type, cur_value, html_results, check_key=None):
    """
    check current results in range[-0.05, 0.05]
    :param job_info
    :param run_machine_type:
    :param cur_value:
    :param html_result:
    :param check_key:
    :return:
    """
    results = bm.ViewJobResult.objects.filter(model_name=job_info["model_name"], 
                                              report_index_id=job_info["index"],
                                              job_type=2,
                                              cuda_version=args.cuda_version,
                                              cudnn_version=args.cudnn_version,
                                              device_type=args.device_type,
                                              model_implement_type=args.implement_type,
                                              frame_name="paddlepaddle",
                                              run_machine_type=run_machine_type).order_by('-version')

    results_list = []
    for result in results:
        if len(results_list) == 3:
            break
        try:
            if result: # json.loads("") is bug
                result = json.loads(result.report_result)
                result = result if isinstance(result, dict) else float(result)
                if result: #check if not zero         
                    results_list.append(result)

        except Exception as e:
            print "add history data error {}".format(e)

    #如果历史数据一直为空，则不报警
    if not results_list:
        return

    try:
        if isinstance(results_list[0], dict) and check_key:
            results_list = [float(x[check_key]) for x in results_list]
            cur_value = float(json.loads(cur_value)[check_key])
        avg_values = round(np.array(results_list).mean(), 4)
        ranges = round((float(cur_value) - avg_values) / avg_values, 4)
    except Exception as rw:
        print "range solve error {}".format(rw)
        ranges = -1
    if ranges > 0.05 or ranges < -0.05:
        current_html_result = [job_info["model_name"], run_machine_type,
                               check_key if check_key else DICT_INDEX[job_info["index"]], 
                               avg_values, cur_value, ranges]
        html_results.append(current_html_result)


def insert_results(job_id, model_name, report_index_id, result, log_path=0):
    """insert job results to db"""
    pjr = bm.JobResults()
    pjr.job_id = job_id
    pjr.model_name = model_name
    pjr.report_index_id = report_index_id
    pjr.report_result = result
    pjr.train_log_path = log_path
    pjr.save()
    return pjr


def insert_job(image_id, run_machine_type, job_info, args):
    """ insert job to db"""
    cluster_job_id = uuid.uuid1()
    pj = bm.Job()
    pj.job_name = "pb_{}_{}".format(args.paddle_version, job_info["model_name"])
    pj.cluster_job_id = cluster_job_id
    pj.cluster_type_id = "LocalJob"
    pj.model_name = job_info["model_name"]
    pj.report_index = job_info["index"]
    pj.code_branch = "master"
    pj.code_commit_id = args.code_commit_id
    pj.job_type = args.job_type
    pj.run_machine_type = run_machine_type
    pj.frame_id = 0
    pj.image_id = image_id
    pj.cuda_version = args.cuda_version
    pj.cudnn_version = args.cudnn_version
    pj.device_type = args.device_type
    pj.model_implement_type = args.implement_type
    pj.log_extracted = "yes"
    pj.save()
    return pj


def parse_logs(args):
    """
    parse log files and insert to db
    :param args:
    :return:
    """
    image_id = get_image_id()
    file_list = load_folder_files(os.path.join(args.log_path, "index"))  
    html_results = []
    for job_file in file_list:
        result = 0
        with open(job_file, 'r+') as file_obj:
            file_lines = file_obj.readlines()
            try:
                job_info = json.loads(file_lines[-1])
            except Exception as exc:
                print("file {} parse error".format(job_file))
                continue

            # save job         
            if str(job_info["gpu_num"]) == "8" and job_info["run_mode"] == "mp":
                run_machine_type = DICT_RUN_MACHINE_TYPE['8mp']
            else:
                run_machine_type = DICT_RUN_MACHINE_TYPE[str(job_info["gpu_num"])]
            job_id = insert_job(image_id, run_machine_type, job_info, args).job_id

            # parse job results
            cpu_utilization_result = 0
            gpu_utilization_result = 0
            try:
                if job_info["index"] == 1:
                    result = job_info['FINAL_RESULT']
                    for line in file_lines:
                        if 'AVG_CPU_USE' in line:
                            cpu_utilization_result = line.strip().split('=')[1]
                        if 'AVG_GPU_USE' in line:
                            gpu_utilization_result = line.strip().split('=')[1]
                elif job_info["index"] == 2:
                    for line in file_lines:
                        if "MAX_GPU_MEMORY_USE" in line:
                            value = line.strip().split("=")[1].strip()
                            result = int(value) if str.isdigit(value) else 0
                            break
                elif job_info["index"] == 3:
                    result = json.dumps(job_info['FINAL_RESULT'])
                else:
                    for line in file_lines:
                        if "MAX_BATCH_SIZE" in line:
                            value = line.strip().split("=")[1].strip()
                            result = int(value) if str.isdigit(value) else 0
                            break

                # save job results
                pjr = insert_results(job_id, job_info["model_name"], job_info["index"], result, 1)
                log_file = job_info["log_file"].split("/")[-1]
                train_log_path = LOG_SERVER + os.path.join(os.path.basename(args.log_path), "train_log", log_file)
                log_save_dict = {"train_log_path": train_log_path}
                if job_info["index"] == 1:
                    insert_results(job_id, job_info["model_name"], 7, cpu_utilization_result)
                    insert_results(job_id, job_info["model_name"], 8, gpu_utilization_result)
                    if int(job_info["gpu_num"]) == 1:
                        profiler_log = job_info["log_with_profiler"].split("/")[-1]
                        profiler_path = job_info["profiler_path"].split("/")[-1]               
                        profiler_log_path = LOG_SERVER + os.path.join(os.path.basename(args.log_path), "profiler_log", profiler_log)
                        profiler_path = LOG_SERVER + os.path.join(os.path.basename(args.log_path), "profiler_log", profiler_path)
                        log_save_dict["profiler_log_path"] = profiler_log_path
                        log_save_dict["profiler_path"] = profiler_path
                
                pjrl = bm.JobResultsLog()
                pjrl.result_id = pjr.result_id
                pjrl.log_path = json.dumps(log_save_dict)
                pjrl.save()
                # cmd = "curl -I -m 10 -o /dev/null -s -w %{http_code} " + profiler_log_path
                #if commands.getoutput(cmd) != '200':

            except Exception as pfe:
                print pfe
            else:
                print("models: {}, run_machine_type: {}, index: {}, result: {}".format(
                    job_info["model_name"], run_machine_type, job_info["index"], result))

                if job_info["index"] != 3:
                    check_results(job_info, run_machine_type, result, html_results)            
                elif job_info["index"] == 3:
                    check_results(job_info, run_machine_type, result, html_results, "Framework_Total")
                    check_results(job_info, run_machine_type, result, html_results, "GpuMemcpy_Total")

    if html_results:
        template.construct_email_content(html_results, args.log_path, args)


if __name__ == '__main__':
    args = parser.parse_args()
    parse_logs(args)
