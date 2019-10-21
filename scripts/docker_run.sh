#!/bin/bash

usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  all
  -d  dir of benchmark_work_path
  -c  cuda_version 9.0|10.0
  -n  cudnn_version 7
  -a  image_branch develop|1.6|pr_number|v1.6.0
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, images such as /ssd1/ljh
  -r  run_module  ce or local
  -t  job_type  benchmark_daliy | models test | pr_test
  -g  device_type  p40 | v100
  -s  implement_type of model static | dynamic
  -e  benchmark alarm email address
EOF
}
if [ $# -lt 18 ] ; then
  usage
  exit 1;
fi
while getopts h:m:d:c:n:a:p:r:t:g:s:e: opt
do
  case $opt in
  h) usage; exit 0 ;;
  m) model="$OPTARG" ;;
  d) benchmark_work_path="$OPTARG" ;;
  c) cuda_version="$OPTARG" ;;
  n) cudnn_version="$OPTARG" ;;
  a) image_branch="$OPTARG" ;;
  p) all_path="$OPTARG" ;;
  r) run_module="$OPTARG" ;;
  t) job_type="$OPTARG" ;;
  g) device_type="$OPTARG" ;;
  s) implement_type="$OPTARG" ;;
  e) email_address="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done

export https_proxy=http://172.19.57.45:3128
export http_proxy=http://172.19.57.45:3128

paddle_repo="https://github.com/PaddlePaddle/Paddle.git"

export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

#build paddle
build(){

    if [ ${run_module} = "local" ]; then
        if [ -e ${benchmark_work_path} ]; then
             rm -rf ${benchmark_work_path}/Paddle
        else
            mkdir -p ${benchmark_work_path}
        fi
        cd ${benchmark_work_path}
        git clone ${paddle_repo}
        cd Paddle
    else
        cd ${benchmark_work_path}/Paddle
    fi

    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id is: "${image_commit_id}

    PADDLE_DEV_NAME=docker.io/paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    #version=`date '+%Y%m%d%H%M%S'`
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    PADDLE_VERSION=${version}'.post'$(echo $cuda_version|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
    image_name=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    echo "image_name is: "${image_name}

    #double check1: In some case, docker would hang while compiling paddle, so to avoid re-compilem, need this
    if [ -e ${all_path}/images/${image_name} ]
    then
        echo "image had built, begin running models"
        return
    else
        echo "image not found, begin building"
    fi

    docker pull ${PADDLE_DEV_NAME}
    nvidia-docker run -i --rm -v $PWD:/paddle \
      -w /paddle \
      -e "CMAKE_BUILD_TYPE=Release" \
      -e "PYTHON_ABI=cp27-cp27mu" \
      -e "PADDLE_VERSION=0.0.0.${PADDLE_VERSION}" \
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
      -e "http_proxy=${http_proxy}" \
      -e "https_proxy=${https_proxy}" \
      ${PADDLE_DEV_NAME} \
       /bin/bash -c "paddle/scripts/paddle_build.sh build"
    mkdir -p ./output

    if [[ -d ${all_path}/images ]]; then
        echo "images dir already exists"
    else
        mkdir -p ${all_path}/images
    fi

    if [[ -d ${all_path}/logs ]]; then
        echo "images dir already exists"
    else
        mkdir -p ${all_path}/logs
    fi

    build_link="${CE_SERVER}/viewLog.html?buildId=${BUILD_ID}&buildTypeId=${BUILD_TYPE_DI}&tab=buildLog"
    echo "build log link: ${build_link}"
    #double check2
    if [[ -s ./build/python/dist/${image_name} ]]
    then
        cp ./build/python/dist/${image_name} ${all_path}/images/
        echo "build paddle success, begin run !"
    else
        echo "build paddle failed, exit!"
        sendmail -t ${email_address} <<EOF
From:paddle_benchmark@baidu.com
SUBJECT:benchmark运行结果报警, 请检查
Content-type: text/plain
PADDLE BUILD FAILED!!
详情请点击: ${build_link}
EOF
        exit 1
    fi
}

run(){
    RUN_IMAGE_NAME=paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
    nvidia-docker run -i --rm \
        -v /home/work:/home/work \
        -v /ssd1:/ssd1 \
        -v /ssd2:/ssd2 \
        -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
        -v /usr/bin/monquery:/usr/bin/monquery \
        --net=host \
        --privileged \
        $RUN_IMAGE_NAME \
        /bin/bash -c "cd ${benchmark_work_path}/baidu/paddle/benchmark/libs/benchmark;
        bash auto_run_paddle.sh -m $model \
        -c ${cuda_version} \
        -n ${all_path}/images/${image_name} \
        -i ${image_commit_id} \
        -a ${image_branch} \
        -v ${PADDLE_VERSION} \
        -p ${all_path} \
        -t ${job_type} \
        -g ${device_type} \
        -s ${implement_type}"
}

send_email(){
    # if [[ ${job_type} == 2 && -e ${all_path}/logs/log_${PADDLE_VERSION}/mail.html ]]; then
    if [[ -e ${all_path}/logs/log_${PADDLE_VERSION}/mail.html ]]; then
        cat ${all_path}/logs/log_${PADDLE_VERSION}/mail.html |sendmail -t ${email_address}
    fi
}

zip_log(){
    echo $(pwd)
    if [[ -d ${all_path}/logs/log_${PADDLE_VERSION} ]]; then
        rm -rf output/*
        tar -zcvf output/log_${PADDLE_VERSION}.tar.gz ${all_path}/logs/log_${PADDLE_VERSION}
        cp ${all_path}/images/${image_name}  output/
    fi
}

build
run
zip_log

if [ ${device_type} == "v100" ]; then
    send_email
fi
