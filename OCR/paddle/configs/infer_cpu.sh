#!/bin/bash

set -x

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

export OCR_mode=CPU # CPU, GPU, MKLDNN
export OCR_model_path=ocr_attention/ocr_attention_params
export OCR_batch_size=1
export OCR_skip_batch_num=10
export OCR_iterations=2000
