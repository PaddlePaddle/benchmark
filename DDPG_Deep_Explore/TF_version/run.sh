#!/bin/bash
cd "$(dirname "$0")"
alias python3='/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib:/home/zhoubo01/tools/miniconda2/envs/opensim-rl/lib /home/zhoubo01/tools/miniconda2/envs/opensim-rl/bin/python'
python3 ./algorithm.py --ensemble_num 1 --test_times 10
