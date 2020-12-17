# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import logging
import argparse
import requests


def parse_log_file(log_file):
    """Load one case result from log file.
    """
    result = dict(file_name=os.path.split(log_file)[-1])

    if not os.path.exists(log_file):
        logging.error("%s is not exists!" % log_file)
        return

    with open(log_file) as f:
        for line in f.read().strip().split('\n')[::-1]:
            try:
                result.update(json.loads(line))
                return result
            except ValueError:
                pass  # do nothing

    # run fail
    return None


def parse_log_result(log_result):
    """Convert log result into benchmark API request.
    """
    if log_result is None:
        return

    request = dict()

    op_case_name, options = log_result.get("file_name").split('-')
    framework, processor, task, direction = options.split('.')[0].split('_')

    speed_result = log_result.get("speed", dict())

    request["name"] = op_case_name
    request["frame"] = framework
    request["version"] = os.getenv("BENCHMARK_VERSION")
    request["commit"] = os.getenv("%s_COMMIT_ID" % framework)
    request["processor"] = dict(cpu=1, gpu=2).get(processor)
    request["timestamp"] = os.getenv("PADDLE_COMMIT_TIME")
    if framework == "paddle":
        request["config"] = log_result.get("parameters")
    if log_result.get("backward"):
        if task == "accuracy":
            request["consistency_backwards"] = log_result.get("consistent",
                                                              "--")
        elif task == "speed":
            request["perf_backwards"] = speed_result.get("total", "--")
            request["gpu_time_backward"] = speed_result.get("gpu_time", "--")
    else:
        if task == "accuracy":
            request["consistency"] = log_result.get("consistent", "--")
        elif task == "speed":
            request["perf"] = speed_result.get("total", "--")
            request["gpu_time"] = speed_result.get("gpu_time", "--")

    return request


def combine_benchmark_request(benchmark_requests, log_request):
    """Combine requests of one case into one request.
    """
    if log_request is None:
        return

    request_key = "%s-%s" % (log_request.get("name"), log_request.get("frame"))

    benchmark_request = benchmark_requests.get(request_key)
    if benchmark_request is None:
        benchmark_requests[request_key] = log_request
    else:
        benchmark_request.update(log_request)


if __name__ == "__main__":
    """Parse daily task logs and write to benchmarck platform.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Specify the log files directory.")
    args = parser.parse_args()

    url = os.getenv("BENCHMARK_URL")
    assert url, "Cant find url for request benchmark platform."

    benchmark_requests = dict()

    for log_file_name in os.listdir(args.logs_dir):
        if log_file_name == "api_info.txt":
            continue
        log_file = os.path.join(args.logs_dir, log_file_name)
        logging.info("Parse file \"%s\"" % log_file_name)
        log_result = parse_log_file(log_file)
        log_request = parse_log_result(log_result)
        combine_benchmark_request(benchmark_requests, log_request)

    args = dict(
        DOCKER_IMAGES=os.getenv("DOCKER_IMAGES"),
        CUDA_VERSION=os.getenv("CUDA_VERSION"),
        CUDNN_VERSION=os.getenv("CUDNN_VERSION"),
        PADDLE_COMMIT_ID=os.getenv("PADDLE_COMMIT_ID"))
    data = benchmark_requests.values()
    r = requests.post(url, dict(args=args, data=data))
    logging.info(r.ok)
