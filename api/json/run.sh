#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

unset GREP_OPTIONS

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

declare -A SPECIAL_DICTS
# Set ignored params
SPECIAL_DICTS=( \
    ["accuracy"]="k correct total" \
    ["affine_channel"]="act" \
    ["anchor_generator"]="anchor_sizes" \
    ["assign"]="output" \
    ["batch_norm"]="epsilon is_test act in_place momentum" \
    ["conv2d"]="act num_filters use_cudnn" \
    ["conv2d_transpose"]="act num_filters use_cudnn" \
    ["depthwise_conv2d"]="act num_filters groups" \
    ["dropout"]="is_test seed dropout_prob" \
    ["embedding"]="padding_idx is_sparse" \
    ["fc"]="size" \
    ["fill_constant"]="value force_cpu" \
    ["instance_norm"]="epsilon" \
    ["l2_normalize"]="epsilon" \
    ["layer_norm"]="act epsilon" \
    ["leaky_relu"]="alpha" \
    ["matmul"]="alpha" \
    ["pool2d"]="use_cudnn" \
    ["reshape"]="shape inplace" \
    ["roi_align"]="spatial_scale" \
    ["scale"]="scale bias act" \
    ["softmax_with_cross_entropy"]="return_softmax ignore_index" \
    ["transpose"]="perm" \
)

declare -A SIMILAR_API
# Set similar APIs
SIMILAR_API=( \
    ["activation"]="abs cos exp floor relu sin sigmoid softsign square sqrt tanh" \
    ["arg"]="argmax argmin" \
    ["compare"]="less_than less_equal not_equal greater_than greater_equal equal" \
    ["elementwise"]="elementwise_add elementwise_div elementwise_max elementwise_min elementwise_mul elementwise_sub elementwise_pow" \
    ["logical"]="logical_and logical_or" \
    ["reduce"]="reduce_mean reduce_sum reduce_prod" \
)

filenames=$(ls ${INPUT_DIR}/*.json)
for file in ${filenames}
do
    op_name=$(basename ${file} .json)

    skipped=0
    for key in $(echo ${!SIMILAR_API[*]})
    do
        result=$(echo ${SIMILAR_API[${key}]} | grep -w "${op_name}")
        if [[ ${result} != "" ]]; then
            skipped=1
            break
        fi
    done
    if [[ ${skipped} -eq 1 ]]; then
        continue
    fi

    append_params=''
    for key in $(echo ${!SPECIAL_DICTS[*]})
    do
        if [[ ${key} = ${op_name} ]]; then
            append_params=${SPECIAL_DICTS[${key}]}
            break
        fi
    done
    echo "processing API:" ${op_name}
    python2 select_configs.py --input_json_file ${file} --output_json_file ${OUTPUT_DIR}/${op_name}.json --ignored_params ${append_params}
    echo ""
done

for key in $(echo ${!SIMILAR_API[*]})
do
    echo "processing API:" ${SIMILAR_API[${key}]}
    python2 select_configs.py --input_json_file ${INPUT_DIR} --output_json_file ${OUTPUT_DIR}/${key}.json --similar_api ${SIMILAR_API[${key}]} --ignored_params "act"
    echo ""
done
