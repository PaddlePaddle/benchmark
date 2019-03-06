export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export FLAGS_cudnn_deterministic=true

#export FLAGS_benchmark=1
#nvprof -o timeline_output_medium -f --cpu-profiling off  --profile-from-start off  python  train.py \
CUDA_VISIBLE_DEVICES='' python  ptb_lm.py large True
