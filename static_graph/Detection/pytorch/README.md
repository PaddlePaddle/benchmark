# PyTorch检测模型

我们使用[facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)测试PyTorch检测模型的性能。
由于[facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)不支持Cascade-RCNN模型，我们使用
[zhaoweicai/Detectron-Cascade-RCNN](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)来测试Cascade-RCNN模型。

## 代码修改
由于[facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)官方不支持coco2017数据集，我们需要做如下修改：

- 使用[./src/dataset_catalog.py](./src/dataset_catalog.py)替换[Detectron/detectron/datasets/dataset_catalog.py](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/dataset_catalog.py)，
该文件的修改如下：

```diff
Detectron(master) $ git diff
diff --git a/detectron/datasets/dataset_catalog.py b/detectron/datasets/dataset_catalog.py
index b92487e..0115493 100644
--- a/detectron/datasets/dataset_catalog.py
+++ b/detectron/datasets/dataset_catalog.py
@@ -37,6 +37,18 @@ _RAW_DIR = 'raw_dir'

 # Available datasets
 _DATASETS = {
+    'coco_2017_train': {
+        _IM_DIR:
+            _DATA_DIR + '/coco/train2017',
+        _ANN_FN:
+            _DATA_DIR + '/coco/annotations/instances_train2017.json'
+    },
+    'coco_2017_valminusminival': {
+        _IM_DIR:
+            _DATA_DIR + '/coco/val2017',
+        _ANN_FN:
+            _DATA_DIR + '/coco/annotations/instances_val2017.json'
+    },
     'cityscapes_fine_instanceonly_seg_train': {
         _IM_DIR:
             _DATA_DIR + '/cityscapes/images',
```

## 准备数据

数据集默认目录为：Detectron/detectron/datasets/data，可通过以下命令将已有数据集软链接过来：
```
$ cd Detectron/detectron/datasets/data
$ mkdir coco
$ cd coco
$ ln -s /data/COCO17/annotations annotations
$ ln -s /data/COCO17/train2017 train2017
$ ln -s /data/COCO17/test2017 test2017
$ ln -s /data/COCO17/val2017 val2017
```

## 配置文件

- Cascade RCNN-FPN模型，使用配置文件[./configs/e2e_cascade_rcnn_R-50-FPN_1x.yaml](./configs/e2e_cascade_rcnn_R-50-FPN_1x.yaml)。该配置文件是基于
基于[e2e_cascade_rcnn_R-50-FPN_1x.yaml](https://github.com/zhaoweicai/Detectron-Cascade-RCNN/blob/master/configs/cascade_rcnn_baselines/e2e_cascade_rcnn_R-50-FPN_1x.yaml)，
做了如下修改：

```diff
cascade_rcnn_baselines(master) $ git diff
diff --git a/configs/cascade_rcnn_baselines/e2e_cascade_rcnn_R-50-FPN_1x.yaml b/configs/cascade_rcnn_baselines/e2e_cascade_rcnn_R-50-FPN_1x.yaml
index ffe9a6b..0c3b4fb 100644
--- a/configs/cascade_rcnn_baselines/e2e_cascade_rcnn_R-50-FPN_1x.yaml
+++ b/configs/cascade_rcnn_baselines/e2e_cascade_rcnn_R-50-FPN_1x.yaml
@@ -5,11 +5,11 @@ MODEL:
   FASTER_RCNN: True
   CASCADE_ON: True
   CLS_AGNOSTIC_BBOX_REG: True  # default: False
-NUM_GPUS: 8
+NUM_GPUS: 1
 SOLVER:
   WEIGHT_DECAY: 0.0001
   LR_POLICY: steps_with_decay
-  BASE_LR: 0.02
+  BASE_LR: 0.002
   GAMMA: 0.1
   MAX_ITER: 90000
   STEPS: [0, 60000, 80000]
@@ -28,8 +28,8 @@ CASCADE_RCNN:
   TEST_STAGE: 3
   TEST_ENSEMBLE: True
 TRAIN:
-  WEIGHTS: https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
-  DATASETS: ('coco_2014_train', 'coco_2014_valminusminival')
+  WEIGHTS: https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
+  DATASETS: ('coco_2017_train', 'coco_2017_valminusminival')
   SCALES: (800,)
   MAX_SIZE: 1333
   BATCH_SIZE_PER_IM: 512
```

