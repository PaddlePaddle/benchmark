set -e # set -o errexit
set -u # set -o nounset
set -o pipefail 

export FLAGS_fraction_of_gpu_memory_to_use=0.001
export CUDA_VISIBLE_DEVICES=0

in_task_type='train'
in_conf_file_path='examples/mmdnn-pointwise.json'
python paddle_simnet.py \
		   --task_type $in_task_type \
		   --conf_file_path $in_conf_file_path

