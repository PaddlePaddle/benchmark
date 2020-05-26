import os
import importlib
import common_import

NOT_API = [
    "conv2d_transpose", "main", "null", "fetch", "common_import", "register",
    "pool2d", "softmax_with_cross_entropy", "launch"
]


def print_api():
    path = os.getcwd()
    for filename in os.listdir(path):
        api_name = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        if file_extension == '.py' and api_name not in NOT_API:
            module = import_api(api_name)
            module.register_api()

    print(common_import.REGISTER_API_INFO)


def import_api(api_name):
    try:
        module = importlib.import_module(api_name)
        #print("Successly import %s" % (api_name))
        return module
    except Exception:
        print("Failed to import %s" % (api_name))


if __name__ == '__main__':
    print_api()
