source activate python35
export CUDA_VISIBLE_DEVICES="1"

#wget ftp://yq01-sys-hic-p40-box-a12-0057.yq01.baidu.com:/home/users/minqiyang/workspace/paddle/Paddle/build935/accelerate_ddpg/python/dist/paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl -O paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl && pip uninstall -y paddlepaddle-gpu && pip install paddlepaddle_gpu-0.0.0-cp35-cp35m-linux_x86_64.whl

export PATH=/usr/local/cuda/bin:$PATH
FLAGS_enforce_when_check_program_=0 GLOG_vmodule=operator=1,computation_op_handle=1 python ./multi_thread_test.py --ensemble_num 1 --test_times 10 >log 2>errorlog

#python timeline.py --profile_path=./profile --timeline_path=./timeline
