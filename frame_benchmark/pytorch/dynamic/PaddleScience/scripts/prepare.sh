# Set directories
export PDSC_DIR=$(cd "$( dirname ${BASH_SOURCE[0]})"; cd ..; pwd)
export TEST_DIR="${PDSC_DIR}"
export TIPC_TEST="ON" # open tipc log in solver.py 
export PYTHONPATH=${PDSC_DIR}

echo -e "\n* [TEST_DIR] is now set : \n" ${TEST_DIR} "\n"
echo -e "\n* [PYTHONPATH] is now set : \n" ${PYTHONPATH} "\n"

source ${TEST_DIR}/scripts/common_func.sh

# Read parameters from [/tipc/config/*/*.txt]
PREPARE_PARAM_FILE=$1
dataline=`cat ${TEST_DIR}$PREPARE_PARAM_FILE`
lines=(${dataline})
download_dataset=$(func_parser_value "${lines[61]}")
python=$(func_parser_value "${lines[2]}")
export pip=$(func_parser_value "${lines[62]}")
workdir=$(func_parser_value "${lines[63]}")
${pip} install --upgrade pip
${pip} install pybind11
${pip} install -r ${TEST_DIR}/models/cylinder_2d/requirements.txt

if [ ${download_dataset} ] ; then
    cd ${PDSC_DIR}${workdir}
    ${python} ${PDSC_DIR}${download_dataset}
fi
