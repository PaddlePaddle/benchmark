#!/usr/bin/env bash

# install env
if [ ${RUN_PLAT} == "local" ]; then
    pip install ${all_path}/other_frame_whls/torch-1.10.0-cp37-cp37m-manylinux1_x86_64.whl
    pip install  ${all_path}/other_frame_whls/torchvision-0.11.1-cp37-cp37m-manylinux1_x86_64.whl
else
    pip install torch torchvision
fi
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls

