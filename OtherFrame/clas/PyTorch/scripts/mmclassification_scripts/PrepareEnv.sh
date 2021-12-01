#!/usr/bin/env bash

# install env
pip3 install torch torchvision
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls
