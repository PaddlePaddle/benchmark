import os
import re
import sys
import json
import importlib

sys.path.append("..")
from tests.common_import import *
from common.special_op_list import NO_BACKEND_API

NOT_API = ["main", "common_import", "launch"]
NO_JSON_API = ["feed", "fetch", "fill_constant", "null"]

API_LIST = []
SUB_CONFIG_LIST = []

REGISTER_API_INFO = {}


def subconfig_api():
    subclass_list = APIConfig.__subclasses__()
    for i in range(len(subclass_list)):
        class_name = subclass_list[i].__name__
        module_name = hump2underline(class_name.replace('Config', ''))
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

        if obj.name in NO_BACKEND_API:
            backend = False
        else:
            backend = True

        for api in api_list:
            REGISTER_API_INFO[api] = [obj.name, json_file, backend]


def config_api():
    CONFIG_LIST = list(set(API_LIST).difference(set(SUB_CONFIG_LIST)))
    CONFIG_LIST.remove('__init__')
    for api in CONFIG_LIST:
        if api in NO_BACKEND_API:
            backend = False
        else:
            backend = True
        if api in NO_JSON_API:
            json_file = None
        else:
            json_file = api + '.json'

        REGISTER_API_INFO[api] = [api, json_file, backend]


def fwrite_api():
    with open("auto_run_info.txt", 'w') as f:
        for api in REGISTER_API_INFO.keys():
            f.writelines(api + ',' + str(REGISTER_API_INFO[api][0]) + ',' +
                         str(REGISTER_API_INFO[api][1]) + ',' + str(
                             REGISTER_API_INFO[api][2]) + '\n')

    with open("support_api_list.txt", 'w') as fo:
        for api in REGISTER_API_INFO.keys():
            fo.writelines(str(api) + '\n')


def import_module():
    path = os.getcwd() + '/../tests/'
    for filename in os.listdir(path):
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


def hump2underline(hunp_str):
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', hunp_str).lower()
    return sub


if __name__ == '__main__':
    import_module()
    subconfig_api()
    config_api()
    fwrite_api()
