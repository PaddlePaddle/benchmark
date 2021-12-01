#!/usr/bin/env bash

# install env
pip install torch torchvision
sed -i "s/opencv-python==3.4.1.15/opencv-python==4.4.0.46/g" requirements.txt
sed -i "s/shapely==1.6.4/shapely/g" requirements.txt
pip install -r requirements.txt
