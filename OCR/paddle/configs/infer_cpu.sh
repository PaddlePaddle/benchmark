#!/bin/bash

set -x

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

export OCR_mode=CPU # CPU, GPU, MKLDNN
export OCR_model_path=gpu_attention_models_512/params_01000
export OCR_batch_size=1
export OCR_skip_batch_num=10
export OCR_iterations=2000
