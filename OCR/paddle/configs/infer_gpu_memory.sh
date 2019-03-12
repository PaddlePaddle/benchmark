#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0

export OCR_mode=GPU # CPU, GPU, MKLDNN
export OCR_model_path=gpu_attention_models_512/params_01000
export OCR_batch_size=1
export OCR_skip_batch_num=10
export OCR_iterations=2000
