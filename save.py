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
    pi.image_type = 2
    pi.create_time = ct
    pi.save()

    pis = bm.Image.objects.filter(image_commit_id=args.image_commit_id).order_by('-create_time')
    if pis:
        return pis[0].image_id


def get_job_id(cluster_job_id):
    """
    :param cluster_job_id:
    :return:
    """
    pjs = bm.Job.objects.filter(cluster_job_id=cluster_job_id)
    if pjs:
        return pjs[0].job_id


def parse_logs(args):
    image_id = get_image_id()
    file_list = load_folder_files(os.path.join(args.log_path, "index"))
    dict_run_machine_type = {
        '1gpus' : 'ONE_GPU',
        '4gpus' : 'FOUR_GPU',
        '8gpus' : 'MULTI_GPU',
        '8gpus8p' : 'MULTI_GPU_MULTI_PROCESS'
    }
    cv_models = ['DeepLab_V3+', 'CycleGAN', 'mask_rcnn', 'SE-ResNeXt50', 'yolov3']
    nlp_models = ['bert', 'paddingrnn_large', 'paddingrnn_small', 'transformer']
    rl_models = ['ddpg_deep_explore']

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
        pj = bm.Job()
        pj.job_name = job_name
        pj.cluster_job_id = cluster_job_id
        pj.cluster_type_id = 0
        pj.model_name = model_name
        pj.report_index = report_index
        pj.code_branch = "master"
        pj.code_commit_id = args.code_commit_id
        pj.job_type = 2
        pj.run_machine_type = run_machine_type
        pj.frame_id = 0
        pj.image_id = image_id
        pj.cuda_version = args.cuda_version
        pj.cudnn_version = args.cudnn_version
        pj.log_extracted = "yes"
        pj.save()

        train_log_name = "{}_{}_{}_{}".format(model_name, "train",
                                        task_index, file_name.split('_')[-1][0])
        train_log_path = os.path.join(os.path.basename(args.log_path),
                                      "train_log", train_log_name)

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


if __name__ == '__main__':
    args = parser.parse_args()
    parse_logs(args)