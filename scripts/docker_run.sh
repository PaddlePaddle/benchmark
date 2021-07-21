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
  -g  device_type  A100 | v100
  -s  implement_type of model static | dynamic | dynamic_to_static
  -e  benchmark alarm email address
  -x whl build tag
  -y runtime tag
EOF
}
if [ $# -lt 18 ] ; then
  usage
  exit 1;
fi
while getopts h:m:d:c:n:a:p:r:t:g:s:e:x:y: opt
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
  x) whl_build_tag="$OPTARG" ;;
  y) runtime_tag="$OPTARG" ;;
  \?) usage; exit 1 ;;
  esac
done


echo "#################### x=$whl_build_tag   y=$runtime_tag"
paddle_repo="https://github.com/PaddlePaddle/Paddle.git"

export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

# Construct the paddle version according to the commit date
function construnct_version(){
    cd ${benchmark_work_path}/Paddle
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id is: "${image_commit_id}
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    python_abi='cp27-cp27mu'
    with_gpu='ON'
    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        PADDLE_VERSION=${version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu="OFF"
        cuda_version="10.0"
        cudnn_version=7
    elif [[ 'dynamic_graph' == ${implement_type} ]] || [[ 'static_graph' == ${implement_type} ]] || [[ 'dynamic_to_static' == ${implement_type} ]]; then
        python_abi='cp37-cp37m'
        PADDLE_VERSION=${version}'.post'$(echo ${cuda_version}|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
    else
        PADDLE_VERSION=${version}'.post'$(echo ${cuda_version} | sed 's/\.//g')${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
    fi
    PADDLE_DEV_NAME=paddlepaddle/paddle_manylinux_devel:${whl_build_tag}
    echo "-----------------build IMAGE_NAME is: ${IMAGE_NAME}"
    echo "-----------------build PADDLE_DEV_NAME is: ${PADDLE_DEV_NAME}"
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

    WEEK_DAY=$(date +%w)
    if [[ $WEEK_DAY -eq 6 || $WEEK_DAY -eq 7 ]];then
        cuda_arch_name="All"
    else
        cuda_arch_name="Auto" # Volta or Ampere
    fi
    echo "------------today is $WEEK_DAY, and cuda_arch_name is $cuda_arch_name"
    
    PADDLE_ROOT=${PWD}
    docker run -i --rm -v ${PADDLE_ROOT}:/paddle \
      -w /paddle \
      --net=host \
      -e "CMAKE_BUILD_TYPE=Release" \
      -e "PYTHON_ABI=${python_abi}" \
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
      -e "CUDA_ARCH_NAME=${cuda_arch_name}" \
      -e "CMAKE_VERBOSE_MAKEFILE=OFF" \
      -e "http_proxy=${HTTP_PROXY}" \
      -e "https_proxy=${HTTP_PROXY}" \
      ${PADDLE_DEV_NAME} \
       /bin/bash -c "paddle/scripts/paddle_build.sh build_only"
     build_name=${IMAGE_NAME}

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
        -d ${cudnn_version} \
        -n ${all_path}/images/${IMAGE_NAME} \
        -i ${image_commit_id} \
        -a ${image_branch} \
        -v ${PADDLE_VERSION} \
        -p ${all_path} \
        -t ${job_type} \
        -g ${device_type} \
        -s ${implement_type}"

    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        RUN_IMAGE_NAME=paddlepaddle/paddle:latest
        docker run -i --rm \
            -v /home:/home \
            -v ${all_path}:${all_path} \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE1=${BENCHMARK_WEBSITE1}" \
            -e "BENCHMARK_WEBSITE2=${BENCHMARK_WEBSITE2}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            -e "RUN_IMAGE_NAME=${RUN_IMAGE_NAME}" \
            -e "START_TIME=$(date "+%Y%m%d")" \
            -e "BENCHMARK_TYPE=${BENCHMARK_TYPE}" \
            -e "BENCHMARK_GRAPH=${BENCHMARK_GRAPH}" \
            -e "DEVICE_TYPE=${device_type}" \
            -e "VERSION_CUDA=${cuda_version}" \
            --net=host \
            --privileged \
            --shm-size=128G \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
    else
        RUN_IMAGE_NAME=paddlepaddle/paddle:${runtime_tag}

        if [ ${device_type} == "A100" ]; then
            nvidia-docker run --runtime=nvidia  --gpus '"capabilities=compute,utility,video"' -i --rm \
            -v /home:/home \
            -v ${all_path}:${all_path} \
            -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE1=${BENCHMARK_WEBSITE1}" \
            -e "BENCHMARK_WEBSITE2=${BENCHMARK_WEBSITE2}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            -e "RUN_IMAGE_NAME=${RUN_IMAGE_NAME}" \
            -e "START_TIME=$(date "+%Y%m%d")" \
            -e "BENCHMARK_TYPE=${BENCHMARK_TYPE}" \
            -e "BENCHMARK_GRAPH=${BENCHMARK_GRAPH}" \
            -e "DEVICE_TYPE=${device_type}" \
            -e "VERSION_CUDA=${cuda_version}" \
            --net=host \
            --privileged \
            --shm-size=128G \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
        else
            nvidia-docker run -i --rm \
            -v /home:/home \
            -v ${all_path}:${all_path} \
            -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE1=${BENCHMARK_WEBSITE1}" \
            -e "BENCHMARK_WEBSITE2=${BENCHMARK_WEBSITE2}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            -e "RUN_IMAGE_NAME=${RUN_IMAGE_NAME}" \
            -e "START_TIME=$(date "+%Y%m%d")" \
            -e "BENCHMARK_TYPE=${BENCHMARK_TYPE}" \
            -e "BENCHMARK_GRAPH=${BENCHMARK_GRAPH}" \
            -e "DEVICE_TYPE=${device_type}" \
            -e "VERSION_CUDA=${cuda_version}" \
            --net=host \
            --privileged \
            --shm-size=128G \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
        fi
    fi
}

#Send alarm email
function send_email(){
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
