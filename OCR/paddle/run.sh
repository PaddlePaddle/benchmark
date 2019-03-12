#!/bin/bash

#set -xe

if [ $# -lt 2 ]; then
  echo "Usage: "
  echo "  sh run.sh task config"
  echo "For example:"
  echo "  sh run.sh train train_gpu_perf_1"
  exit
fi

TASK=$1
CONFIG=$2

if [ ${TASK} == "train" ]; then
  case ${CONFIG} in
    train_gpu_perf_1)
      ;;
    train_gpu_perf_8)
      ;;
    train_gpu_memory_1)
      ;;
    train_gpu_memory_8)
      ;;
    *)
      echo "When task is train, config should be train_gpu_perf_1/8 or train_gpu_memory_1/8"
      exit
  esac
elif [ ${TASK} == "infer" ]; then
  case ${CONFIG} in
    infer_gpu_perf)
      ;;
    infer_gpu_memory)
      ;;
    infer_cpu)
      ;;
    *)
      echo "When task is infer, config should be infer_gpu_perf, infer_gpu_memory or infer_cpu"
      exit
    esac
else
  echo "Task should be either train or infer"
  exit
fi

export OCR_work_root="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." && pwd )"

export OCR_model="attention"
#export OCR_model="crnn_ctc"

if [ ${OCR_model} == "crnn_ctc" ]; then
  INIT_MODEL_PACK_NAME=ocr_ctc
  export OCR_init_model=${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}/ocr_ctc_params
elif [ ${OCR_model} == "attention" ]; then
  INIT_MODEL_PACK_NAME=ocr_attention
  export OCR_init_model=${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}/ocr_attention_params
fi

if [ ! -f ${OCR_init_model} ]; then
  echo ">>> Download preptrained params for ${OCR_model} model ..."
  wget https://paddle-ocr-models.bj.bcebos.com/${INIT_MODEL_PACK_NAME}.zip -O ${OCR_work_root}/paddle/${INIT_MODEL_PACK_NAME}.zip
  unzip ${INIT_MODEL_PACK_NAME}.zip 
fi

source ${OCR_work_root}/download.sh

source configs/${CONFIG}.sh
bash ${TASK}.sh
