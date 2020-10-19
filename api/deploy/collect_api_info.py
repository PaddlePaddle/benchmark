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

import os
import re
import sys
import json
import argparse
import importlib

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)
sys.path.append(os.path.join(package_path, "tests"))
sys.path.append(os.path.join(package_path, "tests_v2"))

from common import api_param, special_op_list


class APITestInfo(object):
    def __init__(self, api_name, py_filename, json_filename):
        self.api_name = api_name
        self.py_filename = py_filename
        self.json_filename = json_filename
        self.has_backward = False if api_name in special_op_list.NO_BACKWARD_OPS else True

    def to_string(self):
        return self.api_name + ',' + self.py_filename + ',' + str(
            self.json_filename) + ',' + str(self.has_backward)


def collect_subclass_dict(test_cases_dict):
    """
    Collect the test cases which declares a subclass of APIConfig.
    """
    subclass_list = api_param.APIConfig.__subclasses__()
    subclass_dict = {}
    for item in subclass_list:
        config = item()
        subclass_dict[config.name] = item.__name__
    return subclass_dict


def import_all_tests(test_module_name):
    test_cases_dict = {}
    special_module_list = ["__init__", "common_import"]

    def _import_api(test_module_name, basename):
        try:
            module = importlib.import_module(test_module_name + "." + basename)
            print("Import {} successfully.".format(module.__name__))
            return module
        except Exception as e:
            print("Failed to import {}: {}".format(basename, e))
            return None

    tests_path = os.path.join(package_path, test_module_name)
    for filename in sorted(os.listdir(tests_path)):
        api_name = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        if file_extension == '.py' and api_name not in special_module_list:
            module = _import_api(test_module_name, api_name)
            if module:
                test_cases_dict[api_name] = module

    return test_cases_dict


def main(args):
    # Need to import all modules first
    test_cases_dict = import_all_tests(args.test_module_name)
    subclass_dict = collect_subclass_dict(test_cases_dict)

    op_test_info_dict = {}
    for py_filename in test_cases_dict.keys():
        if py_filename in subclass_dict.keys():
            # Define an object of special APIConfig.
            module = test_cases_dict[py_filename]
            class_name = subclass_dict[py_filename]
            config = getattr(module, class_name)()

            if hasattr(config, "api_list"):
                api_list = config.api_list.keys()
            else:
                api_list = [config.name]

            json_filename = config.alias_filename(config.name + '.json')

            for api_name in api_list:
                info = APITestInfo(
                    api_name=api_name,
                    py_filename=config.name,
                    json_filename=json_filename)
                op_test_info_dict[api_name] = info
        else:
            json_filename = py_filename + ".json" if py_filename not in [
                "feed", "fetch", "null"
            ] else None
            info = APITestInfo(
                api_name=py_filename,
                py_filename=py_filename,
                json_filename=json_filename)
            op_test_info_dict[py_filename] = info

    # Write to filesystem.
    write_str = ""
    for key in sorted(op_test_info_dict.keys()):
        write_str += op_test_info_dict[key].to_string() + "\n"
    with open(args.info_file, 'w') as f:
        f.writelines(write_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--test_module_name',
        type=str,
        default="tests",
        help='The module_name under benchmark/api (tests|tests_v2).')
    parser.add_argument(
        '--info_file',
        type=str,
        default="api_info.txt",
        help='The file is used to collect API information to automatically run the entire APIs.'
    )
    args = parser.parse_args()
    assert args.test_module_name in [
        "tests", "tests_v2"
    ], "Please set test_module_name to \"tests\" or \"tests_v2\"."
    main(args)
