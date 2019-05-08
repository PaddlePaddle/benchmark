#!/bin/bash

cur_model_list=(CycleGAN deeplab se_resnext50 mask_rcnn bert transformer ddpg_deep_explore paddingrnn yolov3)
usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  ${cur_model_list[@]} | all
  -d  dir of benchmark_path
  -c  cuda_version 9.0|10.0
  -n  cudnn_version 7
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, db.py.., such as /ssd1/ljh
EOF
}
if [ $# != 10 ] ; then
  usage
  exit 1;
fi
while getopts h:m:d:c:n:p: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  d) benchmark_path="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  n) cudnn_version="$OPTARG" ;;
  p) all_path="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
paddle_repo="https://github.com/PaddlePaddle/Paddle.git"


#build paddle
build(){

    if [ -e ${benchmark_path} ]
    then
         rm -rf ${benchmark_path}/Paddle
    else
        mkdir -p ${benchmark_path}
    fi
    cd ${benchmark_path}
    git clone ${paddle_repo}
    cd Paddle
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id is: "${image_commit_id}

    PADDLE_DEV_NAME=docker.io/paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    version=`date '+%Y%m%d'`
    PADDLE_VERSION=${version}'.post'$(echo $cuda_version|cut -c1)${cudnn_version}
    image_name=paddlepaddle_gpu-${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    echo "image_name is: "${image_name}

    docker pull ${PADDLE_DEV_NAME}
    docker run -i --rm -v $PWD:/paddle ${PADDLE_DEV_NAME} \
      rm -rf /paddle/third_party /paddle/build /paddle/output /paddle/python/paddle/fluid/core.so

    nvidia-docker run -i --rm -v $PWD:/paddle \
      -w /paddle \
      -e "CMAKE_BUILD_TYPE=Release" \
      -e "PYTHON_ABI=cp27-cp27mu" \
      -e "PADDLE_VERSION=${PADDLE_VERSION}" \
      -e "WITH_DOC=OFF" \
      -e "WITH_AVX=ON" \
      -e "WITH_GPU=ON" \
      -e "WITH_TEST=OFF" \
      -e "RUN_TEST=OFF" \
      -e "WITH_GOLANG=OFF" \
      -e "WITH_SWIG_PY=ON" \
      -e "WITH_PYTHON=ON" \
      -e "WITH_C_API=OFF" \
      -e "WITH_STYLE_CHECK=OFF" \
      -e "WITH_TESTING=OFF" \
      -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
      -e "WITH_MKL=ON" \
      -e "BUILD_TYPE=Release" \
      -e "WITH_DISTRIBUTE=ON" \
      -e "WITH_FLUID_ONLY=OFF" \
      -e "CMAKE_VERBOSE_MAKEFILE=OFF" \
      -e "http_proxy=http://172.19.57.45:3128" \
      -e "https_proxy=http://172.19.57.45:3128" \
      ${PADDLE_DEV_NAME} \
       /bin/bash -c "paddle/scripts/paddle_build.sh build"
    mkdir -p ./output
    cp ./build/python/dist/${image_name} ${all_path}/images/
}

run(){
    if [ -e ${all_path}/images/${image_name} ]
    then
        echo "build paddle success, begin run !"
    else
        echo "build paddle failed, exit!"
        exit 1
    fi
    
    RUN_IMAGE_NAME=paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
    nvidia-docker run -i --rm \
        -v /home/work:/home/work \
        -v /ssd1:/ssd1 \
        -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
        -v /usr/bin/ibdev2netdev:/usr/bin/ibdev2netdev \
        -v /usr/bin/ib_write_bw:/usr/bin/ib_write_bw \
        -v /usr/bin/ofed_info:/usr/bin/ofed_info \
        -v /etc/libibverbs.d:/etc/libibverbs.d \
        -v /usr/lib64/mlnx_ofed/valgrind:/usr/lib64/mlnx_ofed/valgrind \
        --net=host \
        --privileged \
        --shm-size=30G \
        $RUN_IMAGE_NAME \
        /bin/bash -c "cd ${benchmark_path}; bash auto_run_paddle.sh -m $model -c ${cuda_version} -n ${all_path}/images/${image_name} -i ${image_commit_id}  -p ${all_path}"

}

build
run
