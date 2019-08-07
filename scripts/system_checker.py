from __future__ import print_function

import platform
import os
import utils

class CPUInfo(object):
    def __init__(self):
        self.cpu_name = None
        self.cpu_family = None
        self.sockets = 0
        self.cores_per_socket = 0
        self.physical_cores = 0
        self.virtual_cores = 0
        self.ht = 0
        self.numa_nodes = 0

class SystemChecker(object):
    def __init__(self):
        # System infomation: linux, macos or windows
        self.system = platform.system()
        # Detail information of OS
        self.os_detail = platform.platform()
        self.cpuinfo = CPUInfo()
        self.cpu_envs = {}
  
        self._check_cpuinfo()
        self._check_environment()

    def _check_cpuinfo(self):
        if self.system != "Linux":
            print("Current scenario only support in Linux yet!")
            return

        try:
            cpuinfo_res = open('/proc/cpuinfo')
        except EnvironmentError as e:
            warnings.warn(str(e), UserWarning)
        else:
            physical_ids = []
            core_ids = []
            processors = []
            for line in cpuinfo_res:
                if (line.find("physical id") == 0):
                    physical_ids.append(line.split(":")[1].strip())
                elif (line.find("core id") == 0):
                    core_ids.append(line.split(":")[1].strip())
                elif (line.find("processor") == 0):
                    processors.append(line.split(":")[1].strip())
                elif (line.find("model name") == 0):
                    self.cpuinfo.cpu_name = line.split(":")[1].strip()
    
            self.cpuinfo.sockets = len(list(set(physical_ids)))
            self.cpuinfo.cores_per_socket = len(list(set(core_ids)))
            self.cpuinfo.physical_cores = self.cpuinfo.sockets * self.cpuinfo.cores_per_socket
            self.cpuinfo.virtual_cores = len(list(set(processors)))

        lscpu_res, _ = utils.run_shell("lscpu")
        lscpu_lines = lscpu_res.split("\n")
        for line in lscpu_lines:
            if (line.find("Thread(s) per core") == 0):
                self.cpuinfo.ht = int(line.split(":")[1].strip())
            elif (line.find("CPU family") == 0):
                self.cpuinfo.cpu_family = line.split(":")[1].strip()
            elif (line.find("NUMA node(s)") == 0):
                self.cpuinfo.numa_nodes = int(line.split(":")[1].strip())

        print("========================= Hardware Information =========================")
        print("{:<25s} : {}".format("CPU Name", self.cpuinfo.cpu_name))
        print("{:<25s} : {}".format("CPU Family", self.cpuinfo.cpu_family))
        print("{:<25s} : {}".format("Socket Number", self.cpuinfo.sockets))
        print("{:<25s} : {}".format("Cores Per Socket", self.cpuinfo.cores_per_socket))
        print("{:<25s} : {}".format("Total Physical Cores", self.cpuinfo.physical_cores))
        print("{:<25s} : {}".format("Total Virtual Cores", self.cpuinfo.virtual_cores))
        if self.cpuinfo.ht == 1:
            print("{:25s} : {}".format("Hyper Threading", "OFF"))
            if self.cpuinfo.physical_cores != self.cpuinfo.virtual_cores:
                print("Error: HT logical error")
        elif self.cpuinfo.ht > 1:
            print("{:25s} : {}".format("Hyper Threading", "ON"))
            if self.cpuinfo.physical_cores == self.cpuinfo.virtual_cores:
                print("Error: HT logical error")
        print("{:<25s} : {}".format("NUMA Nodes", self.cpuinfo.numa_nodes))
        if self.cpuinfo.numa_nodes != self.cpuinfo.sockets:
            print("Warning: NUMA node is not enough for the best performance, at least {}".format(self.cpuinfo.sockets))

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
