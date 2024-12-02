import os
from os import path as osp
from shutil import copyfile

print("Repeat the images and labels for training...")

repeat_times = 50

gt_dir = "./gtFine/train/multi_class"
img_dir = "./leftImg8bit/train/multi_class"

for file in os.listdir(gt_dir):
    src_path = osp.join(gt_dir, file)
    base_name = file.split("_gtFine")[0]
    for i in range(repeat_times):
        dst_name = base_name + "_{0:03d}_gtFine_labelTrainIds.png".format(i+1)
        dst_path = osp.join(gt_dir, dst_name)
        copyfile(src_path, dst_path)

for file in os.listdir(img_dir):
    src_path = osp.join(img_dir, file)
    base_name = file.split("_leftImg8bit")[0]
    for i in range(repeat_times):
        dst_name = base_name + "_{0:03d}_leftImg8bit.png".format(i+1)
        dst_path = osp.join(img_dir, dst_name)
        copyfile(src_path, dst_path)