#!/usr/bin/env bash

# install env
pip install torch torchvision
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls
