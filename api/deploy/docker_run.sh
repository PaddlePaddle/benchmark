if [[ -t 1 ]]
then
    YELLOW="$( echo -e "\e[33m" )"
    GREEN="$( echo -e "\e[32m" )"
    RED="$( echo -e "\e[31m" )"
    NORMAL="$( echo -e "\e[0m" )"
fi

function _yellow(){ echo "$YELLOW""$@""$NORMAL";}
function _green(){ echo "$GREEN""$@""$NORMAL";}
function _red(){ echo "$RED""$@""$NORMAL";}

function _message(){ echo "$@" >&2;}
function _warn(){ echo $(_yellow '==> WARN:') "$@" >&2;}
function _info(){ echo $(_green '==> ') "$@" >&2;}
function _error(){ echo $(_red '==> ERROR:') "$@" >&2;}
function _fatal(){ echo $(_red '==> ERROR:') "$@" >&2; exit 1;}


function _print_usage(){
    _message "usage: $0 [options]"
    _message "    -h  optional   Print this help message"
    _message "    -b  dir of benchmark_work_path"
    _message "    -d  device_type  p40 | v100 | cpu"
    _message "    -c  cuda_version 9.0|10.0"
    _message "    -n  cudnn_version 7"
    _message "    -a  image_branch develop | 1.6 | pr"
    _message "    -t  job_type  benchmark_daliy | models test | pr_test"
    _message "    -p  all_path contains dir of images such as /ssd1/ljh"
    _message "    -s  implement_type of model static | dynamic"
    _message "    -e  benchmark alarm email address"
}

function _init_parameters(){
    benchmark_work_path=$(pwd)
    image_branch='develop'
    device_type=v100
    cuda_version=10.1
    cudnn_version=7
    job_type=12
    all_path=/ssd1/ljh
    implement_type=staticgraph
}

_init_parameters

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h|--help) _print_usage; exit 0 ;;
    -b|--benchmark_work_path)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        benchmark_work_path=$2
        shift; shift
        ;;
    -d|--device_type)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        device_type=$2
        shift; shift
        ;;
    -c|--cuda_version)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        cuda_version=$2
        shift; shift
        ;;
    -n|--cudnn_version)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        cudnn_version=$2
        shift; shift
        ;;
    -a|--image_branch)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        image_branch=$2
        shift; shift
        ;;
    -t|--job_type)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        job_type=$2
        shift; shift
        ;;
    -p|--all_path)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        all_path=$2
        shift; shift
        ;;
    -s|--implement_type)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        implement_type=$2
        shift; shift
        ;;
    -e|--email_address)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        email_address=$2
        shift; shift
        ;;
    *)
        _fatal "Unrecongnized option $1"
        ;;
   esac
done

function construnct_version(){
    cd ${benchmark_work_path}/Paddle || exit
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    echo "image_commit_id is: "${image_commit_id}
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        PADDLE_VERSION=${version}".${image_branch//-/_}.gcc82"
        IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu="OFF"
        cuda_version="10.0"
        cudnn_version=7
    else
        PADDLE_VERSION=${version}'.gcc82.post'$(echo ${cuda_version}|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp37-cp37m-linux_x86_64.whl
        with_gpu='ON'
    fi
    PADDLE_DEV_NAME=paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
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

    PADDLE_ROOT=${PWD}
    docker run -i --rm -v ${PADDLE_ROOT}:/paddle \
      -w /paddle \
      --net=host \
      -e "CMAKE_BUILD_TYPE=Release" \
      -e "PYTHON_ABI=cp37-cp37m" \
      -e "PADDLE_VERSION=0.0.0.${PADDLE_VERSION}" \
      -e "WITH_AVX=ON" \
      -e "WITH_GPU=${with_gpu}" \
      -e "WITH_TEST=OFF" \
      -e "RUN_TEST=OFF" \
      -e "WITH_PYTHON=ON" \
      -e "WITH_STYLE_CHECK=OFF" \
      -e "WITH_TESTING=OFF" \
      -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
      -e "WITH_MKL=ON" \
      -e "BUILD_TYPE=Release" \
      -e "WITH_DISTRIBUTE=OFF" \
      -e "CMAKE_VERBOSE_MAKEFILE=OFF" \
      -e "http_proxy=${HTTP_PROXY}" \
      -e "https_proxy=${HTTP_PROXY}" \
      ${PADDLE_DEV_NAME} \
       /bin/bash -c "paddle/scripts/paddle_build.sh build $(nproc)"
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

function run(){
    construnct_version
    # Determine if the whl exists
    if [[ -s ${all_path}/images/${IMAGE_NAME} ]]; then echo "image found"; else exit 1; fi
    logs_dir=${all_path}/logs/logs_op_benchmark/${PADDLE_VERSION}
    mkdir -p ${logs_dir}
    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    RUN_IMAGE_NAME=paddlepaddle/paddle:latest-dev-cuda${cuda_version}-cudnn${cudnn_version}-gcc82
    # CPU任务：export CPU_VISIBLE_DEVICES="0,1,2,3,4"，即并行开启5个CPU任务，用了第0-4个核。 
    run_cmd="rm -rf /usr/local/python2.7.15/bin/python;
             rm -rf /usr/local/python2.7.15/bin/pip;
             ln -s /usr/local/bin/python3.7 /usr/local/python2.7.15/bin/python;
             ln -s /usr/local/bin/pip3.7 /usr/local/python2.7.15/bin/pip;
             python -m pip install --upgrade pip;
             pip install nvidia-ml-py;
             pip install psutil;
             pip install tensorflow==2.3;
             pip uninstall paddlepaddle -y;
             pip uninstall paddlepaddle-gpu -y;
             pip install ${all_path}/images/${IMAGE_NAME};
             cd ${benchmark_work_path}/baidu/paddle/benchmark/api;
             bash deploy/main_control.sh tests tests/configs ${logs_dir} ${CUDA_VISIBLE_DEVICES} gpu;
             bash deploy/main_control.sh tests tests/configs ${logs_dir} ${CPU_VISIBLE_DEVICES} cpu;
             unset http_proxy https_proxy;
             python deploy/post_log.py --server_path ${LOG_SERVER} --file_path ${logs_dir}
             ln -s ${all_path}/env/bin/python /usr/local/bin/mypython;
             export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${all_path}/env/lib/;
             mypython deploy/summary.py ${logs_dir};
            "
    nvidia-docker run ${CUDA_SO} ${DEVICES}  -i --rm \
            -v /home/work:/home/work \
            -v ${all_path}:${all_path} \
            -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
            -v /usr/bin/monquery:/usr/bin/monquery \
            -e "BENCHMARK_WEBSITE1=${BENCHMARK_WEBSITE1}" \
            -e "BENCHMARK_WEBSITE2=${BENCHMARK_WEBSITE2}" \
            -e "http_proxy=${HTTP_PROXY}" \
            -e "https_proxy=${HTTP_PROXY}" \
            -e "PADDLE_VERSION=${PADDLE_VERSION}" \
            -e "RUN_IMAGE_NAME=${RUN_IMAGE_NAME}" \
            -e "PADDLE_COMMIT_ID=${image_commit_id}" \
            -e "START_TIME=$(date "+%Y%m%d")" \
            -e "BENCHMARK_TYPE=${BENCHMARK_TYPE}" \
            -e "BENCHMARK_GRAPH=${BENCHMARK_GRAPH}" \
            --net=host \
            --privileged \
            --shm-size=32G \
            ${RUN_IMAGE_NAME} \
            /bin/bash -c "${run_cmd}"
}

#Send alarm email
function send_email(){
    if [[ -e ${logs_dir}/mail.html ]]; then
        cat ${logs_dir}/mail.html |sendmail -t ${email_address}
    fi
}

build_paddle
run
send_email
