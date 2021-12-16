#!/usr/bin/env bash

# install env
if [ ${RUN_PLAT} == "local" ]; then
    pip install ${all_path}/other_frame_whls/torch-1.10.0-cp37-cp37m-manylinux1_x86_64.whl
    pip install  ${all_path}/other_frame_whls/torchvision-0.11.1-cp37-cp37m-manylinux1_x86_64.whl
else
    pip install torch torchvision
fi
sed -i "s/opencv-python==3.4.1.15/opencv-python==4.4.0.46/g" requirements.txt
sed -i "s/shapely==1.6.4/shapely/g" requirements.txt
pip install -r requirements.txt
