#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0

export OCR_mode=GPU # CPU, GPU, MKLDNN
export OCR_parallel=False # True, False
export OCR_batch_size=512
export OCR_log_period=100
export OCR_save_model_period=1000
export OCR_eval_period=1000
export OCR_total_step=1000

