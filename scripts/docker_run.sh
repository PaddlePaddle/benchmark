#!/bin/bash

function usage () {
  cat <<EOF
  usage: $0 [options]
  -h         optional   Print this help message
  -m  model  all
  -d  dir of benchmark_work_path
  -c  cuda_version 9.0|10.0
  -n  cudnn_version 7
  -a  image_branch develop|1.6|pr_number|v1.6.0
  -p  all_path contains dir of prepare(pretrained models), dataset, logs, images such as /ssd1/ljh
  -r  run_module  build_paddle or run_models
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

paddle_repo="https://github.com/PaddlePaddle/Paddle.git"

export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

# Construct the paddle version according to the commit date
function construnct_version(){
    cd ${benchmark_work_path}/Paddle
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id is: "${image_commit_id}

    PADDLE_DEV_NAME=docker.io/paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        PADDLE_VERSION=${version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu="OFF"
        PADDLE_DEV_NAME=hub.baidubce.com/paddlepaddle/paddle:latest-dev
    else
        PADDLE_VERSION=${version}'.post'$(echo ${cuda_version}|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu='ON'
    fi
    echo "IMAGE_NAME is: "${IMAGE_NAME}
}

#build paddle whl and put it to ${all_path}/images
function build_paddle(){
    construnct_version
    #double check1: In some case, docker would hang while compiling paddle, so to avoid re-compilem, need this
    if [[ -e ${all_path}/images/${IMAGE_NAME} ]]
    then
        echo "image had built, begin running models"
        return
    else
        echo "image not found, begin building"
    fi

    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        docker run -i --rm -v $PWD:/paddle \
          -w /paddle \
          -e "http_proxy=${HTTP_PROXY}" \
          -e "https_proxy=${HTTP_PROXY}" \
          ${PADDLE_DEV_NAME} \
           /bin/bash -c "mkdir -p /paddle/build && cd /paddle/build; pip install protobuf; \
                         cmake .. -DPY_VERSION=2.7 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release;\
                         make -j$(nproc)"
        build_name="paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl"

    else
        nvidia-docker run -i --rm -v $PWD:/paddle \
          -w /paddle \
          -e "CMAKE_BUILD_TYPE=Release" \
          -e "PYTHON_ABI=cp27-cp27mu" \
          -e "PADDLE_VERSION=0.0.0.${PADDLE_VERSION}" \
          -e "WITH_DOC=OFF" \
          -e "WITH_AVX=ON" \
          -e "WITH_GPU=${with_gpu}" \
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
          -e "http_proxy=${HTTP_PROXY}" \
          -e "https_proxy=${HTTP_PROXY}" \
          ${PADDLE_DEV_NAME} \
           /bin/bash -c "paddle/scripts/paddle_build.sh build"
         build_name=${IMAGE_NAME}

    fi

    if [[ -d ${all_path}/images ]]; then
        echo "images dir already exists"
    else
        mkdir -p ${all_path}/images
    fi

    if [[ -d ${all_path}/logs ]]; then
        echo "logs dir already exists"
    else
        mkdir -p ${all_path}/logs
    fi
    mkdir ./output
    build_link="${CE_SERVER}/viewLog.html?buildId=${BUILD_ID}&buildTypeId=${BUILD_TYPE_ID}&tab=buildLog"
    echo "build log link: ${build_link}"

    #double check2
    if [[ -s ./build/python/dist/${build_name} ]]
    then
        cp ./build/python/dist/${build_name} ${all_path}/images/${IMAGE_NAME}
        echo "build paddle success, begin run !"
    else
        echo "build paddle failed, exit!"
        sendmail -t ${email_address} <<EOF
From:paddle_benchmark@test.com
SUBJECT:benchmark运行结果报警, 请检查
Content-type: text/plain
PADDLE BUILD FAILED!!
详情请点击: ${build_link}
EOF
        exit 1
    fi

}

# create containers based on mirror ${RUN_IMAGE_NAME} and run jobs
function run_models(){
    construnct_version
    # Determine if the whl exists
    if [[ -s ${all_path}/images/${IMAGE_NAME} ]]; then echo "image found"; else exit 1; fi
    run_cmd="cd ${benchmark_work_path}/baidu/paddle/benchmark/libs/scripts;
        bash auto_run_paddle.sh -m $model \
        -c ${cuda_version} \
        -n ${all_path}/images/${IMAGE_NAME} \
        -i ${image_commit_id} \
        -a ${image_branch} \
        -v ${PADDLE_VERSION} \
        -p ${all_path} \
        -t ${job_type} \
        -g ${device_type} \
        -s ${implement_type}"

    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        RUN_IMAGE_NAME=hub.baidubce.com/paddlepaddle/paddle:latest

        docker run -i --rm \
        -v /home/work:/home/work \
        -v ${all_path}:${all_path} \
        -v /usr/bin/monquery:/usr/bin/monquery \
        -e "BENCHMARK_WEBSITE=${BENCHMARK_WEBSITE}" \
        -e "http_proxy=${HTTP_PROXY}" \
        -e "https_proxy=${HTTP_PROXY}" \
        --net=host \
        --privileged \
        ${RUN_IMAGE_NAME} \
        /bin/bash -c "${run_cmd}"
    else
        RUN_IMAGE_NAME=paddlepaddle/paddle:latest-gpu-cuda${cuda_version}-cudnn${cudnn_version}
        nvidia-docker run -i --rm \
            --shm-size 16G \
            -v /home/work:/home/work \
            -v ${all_path}:${all_path} \
            -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE=${BENCHMARK_WEBSITE}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            --net=host \
            --privileged \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
    fi
}

#Send alarm email
function send_email(){
    # if [[ ${job_type} == 2 && -e ${all_path}/logs/${PADDLE_VERSION}/mail.html ]]; then
    if [[ -e ${all_path}/logs/${PADDLE_VERSION}/${implement_type}/mail.html ]]; then
        cat ${all_path}/logs/${PADDLE_VERSION}/${implement_type}/mail.html |sendmail -t ${email_address}
    fi
}

# Compressed training log and storage
function zip_log(){
    echo $(pwd)
    if [[ -d ${all_path}/logs/${PADDLE_VERSION} ]]; then
        rm -rf output/*
        tar -zcvf output/${PADDLE_VERSION}.tar.gz ${all_path}/logs/${PADDLE_VERSION}
        cp ${all_path}/images/${IMAGE_NAME}  output/
    fi
}

if [[ ${run_module} = "build_paddle" ]]; then
    build_paddle
else
    run_models
    zip_log
    send_email
fi
