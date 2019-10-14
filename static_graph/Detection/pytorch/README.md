# PyTorch检测模型

我们使用[facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)测试PyTorch检测模型的性能。

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
