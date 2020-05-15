# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

import argparse
import os
import feeder
import json
import sys
import re
import warnings

sys.path.append("..")
from common import utils
from common import api_param


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--task',
        type=str,
        default="speed",
        help='Specify the task: [speed|accuracy]')
    parser.add_argument(
        '--framework',
        type=str,
        default="paddle",
        help='Specify the framework: [paddle|tensorflow|tf|both]')
    parser.add_argument(
        '--json_file', type=str, default=None, help='The file of API params')
    parser.add_argument(
        '--api_name', type=str, default=None, help='The series of API')
    parser.add_argument(
        '--config_id',
        type=int,
        default=None,
        help='Only import params of API from json file in the specified position [0|1|...]'
    )
    parser.add_argument(
        '--check_output',
        type=str2bool,
        default=True,
        help='Whether checking the consistency of outputs [True|False]')
    parser.add_argument(
        '--profiler',
        type=str,
        default="none",
        help='Choose which profiler to use [\"none\"|\"Default\"|\"OpDetail\"|\"AllOpDetail\"|\"nvprof\"]'
    )
    parser.add_argument(
        '--backward',
        type=str2bool,
        default=False,
        help='Whether appending grad ops [True|False]')
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--repeat', type=int, default=1, help='Iterations of Repeat running')
    parser.add_argument(
        '--log_level', type=int, default=0, help='level of logging')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU id when benchmarking for GPU')
    args = parser.parse_args()
    gpu_id = args.gpu_id if args.gpu_id > 0 else 0
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        print("CUDA_VISIBLE_DEVICES is None, set to CUDA_VISIBLE_DEVICES={}".
              format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.task not in ["speed", "accuracy"]:
        raise ValueError("task should be speed, accuracy")
    if args.framework not in ["paddle", "tensorflow", "tf", "both"]:
        raise ValueError("task should be paddle, tensorflow, tf, both")
    return args


def run_paddle(task, obj, args, feed_list=None):
    feed = None
    if feed_list is not None:
        assert len(feed_list) == len(obj.feed_vars)

        feed = {}
        for i in range(len(obj.feed_vars)):
            feed[obj.feed_vars[i].name] = feed_list[i]

    if task == "speed":
        if args.run_with_executor:
            obj.run_with_executor(
                use_gpu=args.use_gpu,
                feed=feed,
                repeat=args.repeat,
                log_level=args.log_level,
                check_output=args.check_output,
                profiler=args.profiler)
        else:
            obj.run_with_core_executor(
                use_gpu=args.use_gpu,
                feed=feed,
                repeat=args.repeat,
                log_level=args.log_level,
                check_output=args.check_output,
                profiler=args.profiler)
        return None
    elif task == "accuracy":
        if feed is None:
            raise ValueError("feed should not be None when checking accuracy.")
        outputs = obj.run_with_executor(
            use_gpu=args.use_gpu, feed=feed, check_output=False)
        return outputs


def run_tensorflow(task, obj, args, feed_list=None):
    feed = None
    if feed_list is not None:
        assert len(feed_list) == len(obj.feed_list)

        feed = {}
        for i in range(len(obj.feed_list)):
            feed[obj.feed_list[i]] = feed_list[i]

    if task == "speed":
        profile = True if args.profiler != "none" else False
        obj.run(use_gpu=args.use_gpu,
                feed=feed,
                repeat=args.repeat,
                log_level=args.log_level,
                check_output=args.check_output,
                profile=profile)
        return None
    elif task == "accuracy":
        if feed is None:
            raise ValueError("feed should not be None when checking accuracy.")
        outputs = obj.run(use_gpu=args.use_gpu, feed=feed, check_output=False)
        return outputs


def copy_feed_spec(config=None):
    if config is None or config.feed_spec is None:
        return None
    if not isinstance(config.feed_spec, list):
        config.feed_spec = [config.feed_spec]

    feed_spec = []
    for feed_item in config.feed_spec:
        item = {}
        for key, value in feed_item.items():
            item[key] = value
        feed_spec.append(item)
    return feed_spec


def test_main(pd_obj=None, tf_obj=None, config=None):
    assert config is not None, "API config must be set."

    args = parse_args()
    if args.json_file is not None:
        # Set the filename to alias config's filename, when there is a alias config.
        filename = config.alias_filename(args.json_file)
        if args.config_id is not None and args.config_id >= 0:
            config.init_from_json(filename, args.config_id)
            if args.api_name != None:
                API_s = args.api_name.split(',')
                for api in API_s:
                    config.api = api
                    test_main_without_json(pd_obj, tf_obj, config)
            else:
                test_main_without_json(pd_obj, tf_obj, config)
        else:
            num_configs = 0
            with open(filename, 'r') as f:
                num_configs = len(json.load(f))
            for config_id in range(0, num_configs):
                config.init_from_json(args.json_file, config_id)
                if args.api_name != None:
                    API_s = args.api_name.split(',')
                    for api in API_s:
                        config.api = api
                        test_main_without_json(pd_obj, tf_obj, config)
                else:
                    test_main_without_json(pd_obj, tf_obj, config)
    else:
        test_main_without_json(pd_obj, tf_obj, config)


def _is_paddle_enabled(args, config):
    if args.task == "accuracy" or args.framework in ["paddle", "both"]:
        return True
    return False


def _is_tensorflow_enabled(args, config):
    if config.run_tf:
        if args.task == "accuracy" or args.framework in [
                "tensorflow", "tf", "both"
        ]:
            return True
    return False


def test_main_without_json(pd_obj=None, tf_obj=None, config=None):
    assert config is not None, "API config must be set."

    args = parse_args()
    config.backward = args.backward
    feed_spec = copy_feed_spec(config)
    feed_list = None
    if _is_paddle_enabled(args, config):
        assert pd_obj is not None, "Paddle object is None."
        print(config)
        pd_obj.name = config.name
        pd_obj.create_program()
        pd_obj.build_program(config=config)
        feed_list = feeder.feed_paddle(pd_obj, feed_spec=feed_spec)
        pd_outputs = run_paddle(args.task, pd_obj, args, feed_list)

    if _is_tensorflow_enabled(args, config):
        assert tf_obj is not None, "TensorFlow object is None."
        tf_config = config.to_tensorflow()
        print(tf_config)
        warnings.simplefilter('always', UserWarning)
        tf_obj.name = tf_config.name
        tf_obj.build_graph(config=tf_config)
        feed_list = feeder.feed_tensorflow(
            tf_obj, feed_list, feed_spec=feed_spec)
        tf_outputs = run_tensorflow(args.task, tf_obj, args, feed_list)

    if args.task == "accuracy":
        if tf_config.run_tf:
            utils.check_outputs(
                pd_outputs, tf_outputs, name=pd_obj.name, atol=config.atol)
        else:
            warnings.simplefilter('always', UserWarning)
            warnings.warn("This config is not supported by TensorFlow.")
