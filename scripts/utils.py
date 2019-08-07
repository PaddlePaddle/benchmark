from __future__ import print_function

import subprocess

def run_shell(command, shell=True):
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=shell)

    exit_code = None
    stdout = ''
    while exit_code is None or line:
        exit_code = p.poll()
        line = p.stdout.readline().decode('utf-8')
        stdout += line

    return stdout, exit_code
