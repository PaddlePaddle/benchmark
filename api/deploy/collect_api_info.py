import os
import re
import sys
import json
import argparse
import importlib

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from tests.common_import import *
from common import special_op_list

NOT_API = ["main", "common_import", "launch"]
NO_JSON_API = ["feed", "fetch", "null"]

API_LIST = []
SUB_CONFIG_LIST = []

REGISTER_API_INFO = {}


def collect_subconfig_info():
    subclass_list = APIConfig.__subclasses__()
    for i in range(len(subclass_list)):
        class_name = subclass_list[i].__name__
        module_name = hump_to_underline(class_name.replace('Config', ''))
        SUB_CONFIG_LIST.append(module_name)

        module = import_api(module_name)
        obj_class_name = getattr(module, class_name)
        obj = obj_class_name()

        if hasattr(obj, "api_list"):
            api_list = obj.api_list.keys()
        else:
            api_list = [obj.name]

        if hasattr(obj, "alias_config"):
            json_file = obj.alias_config.name + '.json'
        else:
            json_file = obj.name + '.json'

        if obj.name in special_op_list.NO_BACKWARD_API:
            backend = False
        else:
            backend = True

        for api in api_list:
            REGISTER_API_INFO[api] = [obj.name, json_file, backend]


def collect_config_info():
    CONFIG_LIST = list(set(API_LIST).difference(set(SUB_CONFIG_LIST)))
    CONFIG_LIST.remove('__init__')
    for api in CONFIG_LIST:
        if api in special_op_list.NO_BACKWARD_API:
            backend = False
        else:
            backend = True
        if api in NO_JSON_API:
            json_file = None
        else:
            json_file = api + '.json'

        REGISTER_API_INFO[api] = [api, json_file, backend]


def write_api_info():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--info_file',
        type=str,
        default=None,
        help='The file is used to collect API information to automatically run the entire APIs.'
    )
    parser.add_argument(
        '--support_api_file',
        type=str,
        default=None,
        help='The file includes all APIs currently supported by the benchmark system.'
    )

    args = parser.parse_args()
    with open(args.info_file, 'w') as f:
        for api in sorted(REGISTER_API_INFO.keys()):
            f.writelines(api + ',' + str(REGISTER_API_INFO[api][0]) + ',' +
                         str(REGISTER_API_INFO[api][1]) + ',' + str(
                             REGISTER_API_INFO[api][2]) + '\n')

    if args.support_api_file:
        with open(args.support_api_file, 'w') as fo:
            for api in REGISTER_API_INFO.keys():
                fo.writelines(str(api) + '\n')


def import_module():
    tests_path = os.path.join(package_path, 'tests')
    for filename in os.listdir(tests_path):
        api_name = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        if file_extension == '.py' and api_name not in NOT_API:
            module = import_api(api_name)


def import_api(api_name):
    try:
        api = "." + api_name
        module = importlib.import_module(api, package='tests')
        module_name = module.__name__.split('.')
        API_LIST.append(module_name[1])
        return module
    except Exception:
        print("Failed to import %s" % (api_name))


def hump_to_underline(hunp_str):
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', hunp_str).lower()
    return sub


if __name__ == '__main__':
    import_module()
    collect_subconfig_info()
    collect_config_info()
    write_api_info()
