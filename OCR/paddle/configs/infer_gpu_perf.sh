#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=1

export OCR_mode=GPU # CPU, GPU, MKLDNN
export OCR_model_path=ocr_attention/ocr_attention_params
export OCR_batch_size=1
export OCR_skip_batch_num=10
export OCR_iterations=2000
