#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export OCR_mode=GPU # CPU, GPU, MKLDNN
export OCR_parallel=True # True, False
export OCR_batch_size=256
export OCR_log_period=100
export OCR_save_model_period=1000
export OCR_eval_period=1000
export OCR_total_step=1000

