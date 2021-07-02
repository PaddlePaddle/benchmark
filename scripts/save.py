#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import argparse
import os
import sys
import time
import uuid
import numpy as np
import template
import socket
import json
import traceback
from collections import OrderedDict

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print sys.path
import models.benchmark_server.helper as helper
from benchmark_server import benchmark_models as bm
import run_task_to_icafe as to_icafe

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
    default="static_graph",
    help="The benchmark model implement method, static_graph | dynamic_graph | dynamic_to_static")

DICT_RUN_MACHINE_TYPE = {'1': 'ONE_GPU',
                         '4': 'FOUR_GPU',
                         '8': 'MULTI_GPU',
                         '8mp': 'MULTI_GPU_MULTI_PROCESS',
                         '16': '16_THREADS',
                         '24': '24_THREADS'}

TABLE_HEADER = ["模型", "运行环境", "指标", "当前值", "标准Benchmark值", "相对标准值波幅", "前5次平均值", "相对前5次值波幅"]
TABLE_PROFILE_HEADER = ["模型", "运行环境", "指标", "当前值", "前5次平均值", "相对前5次值波幅"]
DICT_INDEX = {1: "Speed", 2: "Memory", 3: "Profiler_info", 6: "Max_bs"}
# todo config the log_server port
LOG_SERVER = "http://" + socket.gethostname() + ":8777/"
WAVE_THRESHOLD = 0.05
CHECK_TIMES = 5
# fail model list
FAIL_LIST = []


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
        file_list.sort()

    return file_list


def get_image_id():
    """
    :return:
    """
    cur_time = time.time()
    ct = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cur_time))
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


def compute_results(results_list, check_key, cur_value, index, sign=1):
    """
    compute avg_results, range and color.
    Args:
        results_list(list):[benchmark.models.ViewJobResult]
        check_key(str): if the element of results_list is dict and we want to check some key
        cur_value: the value in current version
        index: check report_index_id
        sign: range * sign
    return:
        avg_values(float):
        ranges(float):
        color(str):
    """
    if not results_list:
        return cur_value, 0, "white"
    try:
        if isinstance(cur_value, dict) and check_key:
            results_list = [float(x[check_key]) for x in results_list]
            cur_value = float(cur_value[check_key])
        avg_value = round(np.array(results_list).mean(), 4)
        if not avg_value:
            return cur_value, 0, "white"
        ranges = round((float(cur_value) - avg_value) / avg_value, 4)
    except Exception as rw:
        print "range solve error {}".format(rw)
        traceback.print_exc()
        ranges = -1
    ranges = ranges * sign
    if ranges >= WAVE_THRESHOLD:
        color = "green"
    elif ranges <= -WAVE_THRESHOLD:
        color = "red"
    elif ranges >= WAVE_THRESHOLD:
        color = "red"
    elif ranges <= -WAVE_THRESHOLD:
        color = "greed"
    else:
        color = "white"

    return avg_value, ranges, color


def check_results(model_name, index, run_machine_type, cur_value, html_results, sign=1,
                  check_key=None,
                  is_profile=False, unit="", outlier=0, icafe_results=[]):
    """
    check current results in range[-0.05, 0.05]
    Args:
        model_name(str):
        index(int):
        run_machine_type(str):
        cur_value(dict or float):
        html_result(dict):
        sign(int):
        check_key(str):
    return:
        benchmark(int)
    """
    # 包括pr需要对比的job_type
    if args.job_type in [1, 2]:
        check_job_type = 2
    else:
        check_job_type = args.job_type
    results = bm.ViewJobResult.objects.filter(
        model_name=model_name, report_index_id=index, job_type=check_job_type, cuda_version=args.cuda_version,
        cudnn_version=args.cudnn_version, device_type=args.device_type, model_implement_type=args.implement_type,
        frame_name="paddlepaddle", run_machine_type=run_machine_type, outlier=0).order_by('-version')

    benchmark_results = bm.ViewJobResult.objects.filter(
        model_name=model_name, report_index_id=index, job_type=check_job_type, cuda_version=args.cuda_version,
        cudnn_version=args.cudnn_version, device_type=args.device_type, model_implement_type=args.implement_type,
        frame_name="paddlepaddle", run_machine_type=run_machine_type, benchmark=1, outlier=0).order_by('-version')

    results_list = []
    count = 0
    benchmark = 0
    for result in results:
        if count == 0:
            count += 1
            continue
        if len(results_list) == CHECK_TIMES:
            break
        try:
            if result:  # json.loads("") throws excetion
                result = json.loads(result.report_result)
                result = result if isinstance(result, dict) else float(result)
                if isinstance(result, dict) and result and result[check_key]:  # check if not zero
                    results_list.append(result)
                elif not isinstance(result, dict) and result:
                    results_list.append(result)
        except Exception as e:
            print "add history data error {}".format(e)

    # 如果历史数据和benchmark结果一直为空，则不报警
    if not results_list and not benchmark_results:
        return benchmark
    benchmark_results = [float(i.report_result) for i in benchmark_results[:1]] if benchmark_results else []
    benchmark_value, benchmark_range, benchmark_color = compute_results(benchmark_results, check_key,
                                                                        cur_value, index, sign)
    print('benchmark_value:{}'.format(benchmark_value))
    print('current_value:{}'.format(cur_value))
    if not isinstance(benchmark_value, dict):
        if float(cur_value) < float(benchmark_value) and sign == -1 and outlier == 0:
            benchmark = 1
        elif float(cur_value) > float(benchmark_value) and sign == 1 and outlier == 0:
            benchmark = 1
    avg_value, avg_range, avg_color = compute_results(results_list, check_key, cur_value, index, sign)

    if abs(avg_range) < WAVE_THRESHOLD and abs(benchmark_range) < WAVE_THRESHOLD:
        return benchmark
    print_machine_type = machine_type_to_print(run_machine_type)
    # show detail info only when job success
    if not check_fail(model_name, print_machine_type):
        if is_profile:
            current_html_result = [dict(value=model_name), dict(value=print_machine_type),
                                   dict(value=check_key if check_key else DICT_INDEX[index]),
                                   dict(value="{:.4f}".format(cur_value[check_key]) if check_key
                                   else "{:.4f}".format(cur_value)),
                                   dict(value="{:.4f}".format(avg_value)),
                                   dict(value="{:.2f}%".format(round(avg_range * 100, 2)), color=avg_color)]
        else:
            current_html_result = [dict(value=model_name), dict(value=print_machine_type),
                                   dict(value=check_key if check_key else DICT_INDEX[index]),
                                   dict(value="{:.4f}{}".format(cur_value[check_key], unit) if check_key
                                   else "{:.4f}{}".format(cur_value, unit)),
                                   dict(value="{:.4f}{}".format(benchmark_value[check_key], unit)
                                   if check_key else "{:.4f}{}".format(benchmark_value, unit)),
                                   dict(value="{:.2f}%".format(round(benchmark_range * 100, 2)), color=benchmark_color),
                                   dict(value="{:.4f}{}".format(avg_value, unit)),
                                   dict(value="{:.2f}%".format(round(avg_range * 100, 2)), color=avg_color)]
            if benchmark_color == 'red' and index == 1:
                current_icafe_result = [model_name, print_machine_type, 'down', current_html_result]
                icafe_results.append(current_icafe_result)

        html_results[DICT_INDEX[index]]["data"].append(current_html_result)
    return benchmark


def insert_results(job_id, model_name, report_index_id, result, unit, log_path=0, benchmark=0, outlier=0):
    """insert job results to db"""
    pjr = bm.JobResults()
    pjr.job_id = job_id
    pjr.model_name = model_name
    pjr.report_index_id = report_index_id
    pjr.report_result = result
    pjr.unit = unit
    pjr.outlier = outlier
    pjr.benchmark = benchmark
    pjr.train_log_path = log_path
    pjr.save()
    return pjr


def get_or_insert_model(model_name, mission_name, direction_id):
    """
    根据model_name, 获取mission_id, 如果不存在就创建一个。
    """
    models = bm.BenchmarkModel.objects.filter(model_name=model_name)
    if models:
        return
    missions = bm.Mission.objects.filter(mission_name=mission_name)
    if missions:
        mission_id = missions[0].mission_id
    else:
        ms = bm.Mission()
        ms.mission_name = mission_name
        ms.direction_id = direction_id
        ms.save()
        mission_id = ms.mission_id
    bms = bm.BenchmarkModel()
    bms.model_name = model_name
    bms.mission_id = mission_id
    bms.save()


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


def check_fail(model_name, print_machine_type):
    """
    check whether the specific job is failed, if fail returns True
    """
    for job in FAIL_LIST:
        if model_name == job[0] and print_machine_type == job[1]:
            return True
    return False


def machine_type_to_print(run_machine_type):
    """
    change machine type to print style
    """
    if run_machine_type == 'ONE_GPU':
        if os.getenv("BENCHMARK_TYPE") == 'CPU_Benchmark':
            print_machine_type = '1_THREAD'
        else:
            print_machine_type = '1_GPU'
    elif run_machine_type == 'FOUR_GPU':
        if os.getenv("BENCHMARK_TYPE") == 'CPU_Benchmark':
            print_machine_type = '4_THREADS'
        else:
            print_machine_type = '4_GPUS'
    elif run_machine_type == 'MULTI_GPU':
        if os.getenv("BENCHMARK_TYPE") == 'CPU_Benchmark':
            print_machine_type = '8_THREADS'
        else:
            print_machine_type = '8_GPUS'
    elif run_machine_type == 'MULTI_GPU_MULTI_PROCESS':
        print_machine_type = '8_GPUS_8_PROCESSES'
    else:
        print_machine_type = run_machine_type
    return print_machine_type


def parse_logs(args):
    """
    parse log files and insert to db
    :param args:
    :return:
    """
    image_id = get_image_id()
    file_list = load_folder_files(os.path.join(args.log_path, "index"))
    html_results = OrderedDict()
    icafe_results = []
    for k in DICT_INDEX.values():
        html_results[k] = {}
        if k == 'Profiler_info':
            html_results[k]["header"] = TABLE_PROFILE_HEADER
        else:
            html_results[k]["header"] = TABLE_HEADER
        html_results[k]["data"] = []
    for job_file in file_list:
        result = 0
        with open(job_file, 'r+') as file_obj:
            file_lines = file_obj.readlines()
            try:
                job_info = json.loads(file_lines[-1])
            except Exception as exc:
                print("file {} parse error".format(job_file))
                continue

            # check model if exist in db
            get_or_insert_model(job_info["model_name"], job_info["mission_name"], job_info["direction_id"])

            # save job
            if str(job_info["gpu_num"]) == "8" and job_info["run_mode"] == "mp":
                run_machine_type = DICT_RUN_MACHINE_TYPE['8mp']
            else:
                run_machine_type = DICT_RUN_MACHINE_TYPE[str(job_info["gpu_num"])]
            job_id = insert_job(image_id, run_machine_type, job_info, args).job_id

            # parse job results
            cpu_utilization_result = 0
            gpu_utilization_result = 0
            unit = ''
            outlier = 0
            outlier_mem = 0 
            mem_result = 0
            benchmark = 0
            benchmark_mem = 0
            if job_info["index"] == 1:
                result = job_info['FINAL_RESULT']
                unit = job_info['UNIT']
                fail_flag = job_info['JOB_FAIL_FLAG']
                for line in file_lines:
                    if 'AVG_CPU_USE' in line:
                        cpu_utilization_result = line.strip().split('=')[1]
                    if 'AVG_GPU_USE' in line:
                        gpu_utilization_result = line.strip().split('=')[1]
                    if "MAX_GPU_MEMORY_USE" in line:
                        value = line.strip().split("=")[1].strip()
                        mem_result = int(value) if str.isdigit(value) else 0

            elif job_info["index"] == 3:
                result = json.dumps(job_info['FINAL_RESULT'])
            else:
                for line in file_lines:
                    if "MAX_BATCH_SIZE" in line:
                        value = line.strip().split("=")[1].strip()
                        result = int(value) if str.isdigit(value) else 0
                        break

            print("models: {}, run_machine_type: {}, index: {}, result: {}".format(
                job_info["model_name"], run_machine_type, job_info["index"], result))
            # check_results and send alarm email
            if job_info["index"] == 1:  # speed
                print_machine_type = machine_type_to_print(run_machine_type)
                #record fail jobs
                print('fail_flag:{}'.format(fail_flag))
                if float(result) == 0 or fail_flag == 1:
                    FAIL_LIST.append([job_info["model_name"], print_machine_type])
                    outlier = 1
                    outlier_mem = 1
                    icafe_results.append([job_info["model_name"], print_machine_type, 'fail', []])
                benchmark = check_results(job_info["model_name"], job_info["index"],
                                          run_machine_type, result,
                                          html_results,
                                          -1 if args.device_type.lower() == 'cpu' else 1,
                                          unit=unit, outlier=outlier, icafe_results=icafe_results)
                benchmark_mem = check_results(job_info["model_name"], 2, run_machine_type,
                                              mem_result, html_results,
                                              -1, outlier=outlier_mem,
                                              icafe_results=icafe_results)  # mem
            elif job_info["index"] == 3:  # profiler
                check_results(job_info["model_name"], job_info["index"], run_machine_type,
                              json.loads(result),
                              html_results, -1, "Framework_Total", is_profile=True)
                check_results(job_info["model_name"], job_info["index"], run_machine_type,
                              json.loads(result),
                              html_results, -1, "GpuMemcpy_Total", is_profile=True)
            elif job_info["index"] == 6:  # max BS
                check_results(job_info["model_name"], job_info["index"], run_machine_type,
                              result, html_results, 1)
            else:
                print("--------------> please set a correct index(1|3|6)!")

            try:
                # save job results
                pjr = insert_results(job_id, job_info["model_name"], job_info["index"], result, unit, 1,
                                     benchmark=benchmark, outlier=outlier)
                log_file = job_info["log_file"].split("/")[-1]
                log_base = args.paddle_version + "/" + args.implement_type
                train_log_path = LOG_SERVER + os.path.join(log_base, "train_log", log_file)
                log_save_dict = {"train_log_path": train_log_path}
                if job_info["index"] == 1:
                    insert_results(job_id, job_info["model_name"], 7, cpu_utilization_result, '%')
                    insert_results(job_id, job_info["model_name"], 8, gpu_utilization_result, '%')
                    pjr2 = insert_results(job_id, job_info["model_name"], 2, mem_result, 'MiB', 1,
                                          benchmark=benchmark_mem, outlier=outlier_mem)
                    bm.JobResultsLog.objects.create(
                        result_id=pjr2.result_id, log_path=json.dumps(log_save_dict)).save()
                    if int(job_info["gpu_num"]) == 1:
                        profiler_log = job_info["log_with_profiler"].split("/")[-1]
                        profiler_path = job_info["profiler_path"].split("/")[-1]
                        profiler_log_path = LOG_SERVER + os.path.join(log_base, "profiler_log", profiler_log)
                        profiler_path = LOG_SERVER + os.path.join(log_base, "profiler_log", profiler_path)
                        log_save_dict["profiler_log_path"] = profiler_log_path
                        log_save_dict["profiler_path"] = profiler_path

                bm.JobResultsLog.objects.create(result_id=pjr.result_id, log_path=json.dumps(log_save_dict)).save()
            except Exception as pfe:
                print pfe

    # generate email file
    title = "frame_benchmark"
    env = dict(paddle_branch=args.image_branch, paddle_commit_id=args.image_commit_id,
               benchmark_commit_id=args.code_commit_id, device_type=args.device_type,
               implement_type=args.implement_type, docker_images=os.getenv('RUN_IMAGE_NAME'))
    if args.device_type.upper() in ("P40", "V100", "A100"): 
        env["cuda_version"] = args.cuda_version
        env["cudnn_version"] = args.cudnn_version
    email_t = template.EmailTemplate(title, env, html_results, args.log_path, FAIL_LIST)
    email_t.construct_email_content()
    print('icafe_results:{}'.format(icafe_results))
    # build icafe card
    item = to_icafe.get_alarm_content(icafe_results, env, TABLE_HEADER)
    to_icafe.write_icafe(item)


if __name__ == '__main__':
    args = parser.parse_args()
    parse_logs(args)
