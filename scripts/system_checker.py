from __future__ import print_function

import platform
import os


class SystemChecker(object):
    def __init__(self):
        # System infomation: linux, macos or windows
        self.system = platform.system()
        # Detail information of OS
        self.os_detail = platform.platform()
        self.cpu_envs = {}
  
        self._check_environment()

    def _check_environment(self):
        cpu_env_list = ["KMP_AFFINITY",
                        "OMP_DYNAMIC",
                        "OMP_NESTED",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "MKL_DYNAMIC"]
        for key in cpu_env_list:
            if os.environ.get(key, None) is None:
                self.cpu_envs[key] = "unset"
            else:
                self.cpu_envs[key] = str(os.environ.get(key, None))

        print("------------------ Environment Variables Information -------------------")
        for key in cpu_env_list:
            print("{:<25s} : {}".format(key, self.cpu_envs[key]))


if __name__ == "__main__":
    checker = SystemChecker()
