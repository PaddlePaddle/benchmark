
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_eager_delete_tensor_gb=0.0

workspace_size_limit=256
if [ $# -ge 1 ]; then
  workspace_size_limit=$1
fi

echo "export FLAGS_conv_workspace_size_limit=$workspace_size_limit"
export FLAGS_conv_workspace_size_limit=$workspace_size_limit

python train.py
