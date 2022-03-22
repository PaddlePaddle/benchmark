#!/bin/bash
PWD=`pwd`
python3.8 ./deploy/run_phi_op.py --base_dir $PWD --log_dir logs
cd deploy
python3.8 summary.py ../logs --dump_to_excel True --dump_to_mysql False