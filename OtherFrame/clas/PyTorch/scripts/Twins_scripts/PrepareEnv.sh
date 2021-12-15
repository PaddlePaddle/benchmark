#!/usr/bin/env bash

# install env
pip install torch torchvision
if [ $? -eq 1 ]; then
    pip install ${all_path}/other_frame_whls/torch-1.10.0-cp37-cp37m-manylinux1_x86_64.whl
    pip install  ${all_path}/other_frame_whls/torchvision-0.11.1-cp37-cp37m-manylinux1_x86_64.whl
fi
pip install timm==0.4.5
