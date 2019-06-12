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
    "--gpu_type",
    type=str,
    default='v100',
    help="The benchmark run on v100 or p40")

parser.add_argument(
    "--implement_type",
    type=str,
    default="staticgraph",
    help="The benchmark model implement method")

# log_server_cuda10 = "http://yq01-gpu-255-125-19-00.epc.baidu.com:8887/"
# log_server_cuda9 = "http://yq01-gpu-255-125-21-00.epc.baidu.com:8887/"

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
    # paddle_version = time.strftime('%Y%m%d%H%M%S',
    #                 time.localtime(cur_time)) + '.post{}7'.format(args.cuda_version.split('`,.')[0])

    pi = bm.Image()
    pi.frame_id = 0
    pi.version = args.paddle_version
    pi.cuda_version = args.cuda_version
    pi.cudnn_version = args.cudnn_version
    pi.image_commit_id = args.image_commit_id
    pi.image_type = args.job_type
    pi.create_time = ct
    pi.save()

    pis = bm.Image.objects.filter(image_commit_id=args.image_commit_id).order_by('-create_time')
    if pis:
        return pis[0].image_id


def send_email(title, mailto, cc, content):
    """send email"""
    try:
        # mailto = "liangjinhua01@baidu.com"
        # cc = "liangjinhua01@baidu.com"
        # title = "test for ljh"
        # content = "Model Cyclegan speed down"

        p = subprocess.Popen(['mail', '-s', title, '-c', cc, mailto],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
            )
        out, err = p.communicate(input=content.encode('utf8'))
        print out, err
    except Exception as e:
        print e


def check_results(model_name, index, run_machine_type, cur_value):
    """
    check current results in range[-0.03, 0.03]
    :param model_name:
    :param index:
    :param run_machine_type:
    :param cur_value:
    :return:
    """
    # images = bm.Image.objects.filter(image_type=args.job_type,
    #                                   cuda_version=args.cuda_version,
    #                                   cudnn_version=args.cudnn_version,
    #                                   gpu_type=args.gpu_type,
    #                                   model_implement_type=args.implement_type,
    #                                   frame_id=0).order_by('-version')
    # images_list = []
    # # search top k images
    # for image in images:
    #     if len(images_list) == 3:
    #         break
    #     images_list.append(str(image.version))

    #print images_list
    results = bm.ViewJobResult.objects.filter(model_name=model_name,
                                              report_index_id=index,
                                              job_type=args.job_type,
                                              cuda_version=args.cuda_version,
                                              cudnn_version=args.cudnn_version,
                                              gpu_type=args.gpu_type,
                                              model_implement_type=args.implement_type,
                                              frame_name="paddlepaddle",
                                              run_machine_type=run_machine_type).order_by('-version')

    results_list = []
    for result in results:
        if len(results_list) == 3:
            break
        try:
            if not float(result.report_result) or result.report_result == '-inf':
                continue
            results_list.append(float(result.report_result))
        except Exception as e:
            print "add hitory data error {}".format(e)

    #如果历史数据一直为空，则不报警
    if not results_list:
        return 0

    try:
        avg_values = round(np.array(results_list).mean(), 4)

        try:
            ranges = round((float(cur_value) - avg_values) / avg_values, 4)
        except RuntimeWarning as rw:
            print "range solve error {}".format(rw)
            ranges = -1

        if ranges > 0.05 or ranges < -0.05:
            return avg_values, ranges
        else:
            return 0

    except Exception as e:
        print cur_value, e
        return 0


def get_job_id(cluster_job_id):
    """
    :param cluster_job_id:
    :return:
    """
    pjs = bm.Job.objects.filter(cluster_job_id=cluster_job_id)
    if pjs:
        return pjs[0].job_id


def parse_logs(args):
    """
    parse log files and insert to db
    :param args:
    :return:
    """
    image_id = get_image_id()
    file_list = load_folder_files(os.path.join(args.log_path, "index"))
    dict_run_machine_type = {
        '1gpus' : 'ONE_GPU',
        '4gpus' : 'FOUR_GPU',
        '8gpus' : 'MULTI_GPU',
        '8gpus8p' : 'MULTI_GPU_MULTI_PROCESS'
    }
    cv_models = ['DeepLab_V3+', 'CycleGAN', 'mask_rcnn', 'SE-ResNeXt50', 'yolov3']
    # nlp_models = ['bert', 'paddingrnn_large', 'paddingrnn_small', 'transformer']
    # rl_models = ['ddpg_deep_explore']
    multi_process_models = ['mask_rcnn', 'yolov3', 'transformer', 'bert']
    html_results = []
    for file in file_list:
        # file_name like CycleGAN_mem_1gpus or ddpg_deep_explore_speed_1gpus
        cluster_job_id = uuid.uuid1()
        file_name = file.split('/')[-1]
        model_name = '_'.join(file_name.split('_')[:-2])
        key_word = "FPS:" if model_name in cv_models else 'Avg:'
        job_name = 'pb_' + model_name
        task_index = file_name.split('_')[-2]
        if task_index == 'speed':
            report_index = 1
        elif task_index == 'mem':
            report_index = 2
        else:
            report_index = 6

        run_machine_type = dict_run_machine_type[file_name.split('_')[-1]]
        run_mode = "mp" if file_name.split('_')[-1] == "8gpus8p" else "sp"
        pj = bm.Job()
        pj.job_name = job_name
        pj.cluster_job_id = cluster_job_id
        pj.cluster_type_id = 0
        pj.model_name = model_name
        pj.report_index = report_index
        pj.code_branch = "master"
        pj.code_commit_id = args.code_commit_id
        pj.job_type = args.job_type
        pj.run_machine_type = run_machine_type
        pj.frame_id = 0
        pj.image_id = image_id
        pj.cuda_version = args.cuda_version
        pj.cudnn_version = args.cudnn_version
        pj.gpu_type = args.gpu_type
        pj.model_implement_type = args.implement_type
        pj.log_extracted = "yes"
        pj.save()
        #log_server = log_server_cuda9 if args.cuda_version == '9.0' else log_server_cuda10
        log_server = socket.gethostname()
        #todo config the log_server port
        log_server = "http://" + log_server + ":8777/"
        train_log_name = "{}_{}_{}_{}".format(model_name, "train",
                                               task_index,
                                               file_name.split('_')[-1][0])
        if model_name in multi_process_models:
            train_log_name += "_{}".format(run_mode)
        train_log_path = os.path.join(os.path.basename(args.log_path),
                                      "train_log", train_log_name)
        train_log_path = log_server + train_log_path

        job_id = get_job_id(cluster_job_id)

        result = ""
        with open(file, 'r+') as file_obj:
            file_lines = file_obj.readlines()
            try:
                if report_index == 2:
                    value = file_lines[-1].split()[-1]
                    result = int(value) if str.isdigit(value) else 0
                elif report_index == 1:
                    lines = file_lines[-10:-1]
                    for line in lines:
                        if key_word in line:
                            result = line.split(':')[1].split(' ')[1]
                else:
                    value = file_lines[-1].split()[-1]
                    result = int(value) if str.isdigit(value) else 0

                pjr = bm.JobResults()
                pjr.job_id = job_id
                pjr.model_name = model_name
                pjr.report_index_id = report_index
                pjr.report_result = result
                pjr.train_log_path = train_log_path
                pjr.save()
            except Exception as pfe:
                print pfe
            else:
                print("models: {}, run_machine_type: {}, index: {}, result: {}".format(
                    model_name, run_machine_type, task_index, result))

                # 如果当前值是空或者inf(speed 会出现)
                if not result or result == '-inf':
                    result = 0

                value = check_results(model_name, report_index, run_machine_type, result)

                if value:
                    current_html_result = [model_name, run_machine_type,
                                           task_index, value[0], result, value[1]]
                    html_results.append(current_html_result)

    if html_results:
        template.construct_email_content(html_results, args.log_path, args)


if __name__ == '__main__':
    args = parser.parse_args()
    parse_logs(args)