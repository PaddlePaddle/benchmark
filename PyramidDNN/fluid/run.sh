export PATH=/opt/_internal/cpython-2.7.11-ucs4/bin/:$PATH
export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib/:$LD_LIBRARY_PATH
#ps aux | grep fluid_train | grep -v grep | awk '{print $2}' | xargs kill -9
#ps aux | grep run.sh | awk '{print $2}' | -xargs kill -9

#pip uninstall -y paddlepaddle && pip install paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl

# GLOG_vmodule=fused_hash_embedding_seq_pool_op=2 GLOG_logtostderr=1 OPENBLAS_NUM_THREADS=4 OMP_THREAD_NUM=26 THREAD_NUM=1 CPU_NUM=1 python fluid_train.py
FLAGS_initial_cpu_memory_in_mb=500 FLAGS_reader_queue_speed_test_mode=0 FLAGS_pe_profile_fname=./gperf_output GLOG_v=1 GLOG_logtostderr=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_NUM=1 THREAD_NUM=10 CPU_NUM=1 python fluid_train.py
