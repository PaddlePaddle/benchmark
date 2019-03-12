#!/bin/bash

set -x

DOWNLOAD_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/" && pwd )"

export OCR_train_images=${DOWNLOAD_ROOT}/data/train_images
export OCR_train_list=${DOWNLOAD_ROOT}/data/train.list
export OCR_test_images=${DOWNLOAD_ROOT}/data/test_images
export OCR_test_list=${DOWNLOAD_ROOT}/data/test.list
if [ ! -f ${OCR_train_list} ]; then
  echo ">>> Download data ..."
  wget http://paddle-ocr-data.bj.bcebos.com/data.tar.gz -O ${DOWNLOAD_ROOT}/data.tar.gz
  tar xf ${DOWNLOAD_ROOT}/data.tar.gz -C ${DOWNLOAD_ROOT}
fi

export OCR_TF_train_list=${DOWNLOAD_ROOT}/data/train_tf.list
if [ ! -f ${OCR_TF_train_list} ]; then
  echo ">>> Prepare train list for tf ..."
  cat ${OCR_train_list} | awk '{print $1,$2,image_dir$3,$4;}' image_dir="${OCR_train_images}/" > ${OCR_TF_train_list}
fi

export OCR_TF_test_list=${DOWNLOAD_ROOT}/data/test_tf.list
if [ ! -f ${OCR_TF_test_list} ]; then
  echo "Prepare test list for tf ..."
  cat ${OCR_test_list} | awk '{print $1,$2,image_dir$3,$4;}' image_dir="${OCR_test_images}/" > ${OCR_TF_test_list}
fi
