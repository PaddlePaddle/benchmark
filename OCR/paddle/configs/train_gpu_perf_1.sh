#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=1

export OCR_mode=GPU # CPU, GPU, MKLDNN
export OCR_parallel=False # True, False
export OCR_batch_size=32
export OCR_log_period=100
export OCR_save_model_period=1000
export OCR_eval_period=1000
export OCR_total_step=1000

