镜像是基于https://hub.docker.com/r/pytorch/pytorch/tags 上的 0.4.1-cuda9-cudnn7-runtime建立的


1.重新安装conda 

mkdir conda
cd conda

apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh


2. 准备环境

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

conda install ipython

pip install ninja yacs cython matplotlib tqdm

conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

来自：https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md

3. 安装copoapi
这里安装在：/ssd1/ljh/benchmark/Mask-RCNN/mask-rcc-fb_workspace
  cd benchmark/Mask-RCNN/mask-rcc-fb_workspace
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

4.配置CUDA_HOME
find / -name "libnvblas.so.*"
找到的路径如： /home/work/docker/devicemapper/mnt/294816e447998c28557c2bca88ff5ee1573dac5dd6f51e0b875ea463ceeeaec2/rootfs/usr/local/cuda-9.0/
因为是安装到了conda环境里，创建一个短路径
ln -s /home/work/docker/devicemapper/mnt/294816e447998c28557c2bca88ff5ee1573dac5dd6f51e0b875ea463ceeeaec2/rootfs/usr/local/cuda-9.0/ /usr/local/cuda-9.0/

加入到环境变量export CUDA_HOME=/usr/local/cuda-9.0/


4. 编译maskrcnn-benchmark
  cd ../../maskrcnn-from-fb
  python setup.py build develop
  
链接coco数据地址
mkdir -p datasets/coco
ln -s /ssd1/ljh/dataset/COCO17/annotations datasets/coco/annotations
ln -s /ssd1/ljh/dataset/COCO17/train2017 datasets/coco/train2017
ln -s /ssd1/ljh/dataset/COCO17/test2017 datasets/coco/test2017
ln -s /ssd1/ljh/dataset/COCO17/val2017 datasets/coco/val2017


5. 修改configs/e2e_mask_rcnn_R_50_C4_1x.yaml

 coco2017里：
  5.1 coco_2014_valminusminival 改为 coco_2017_train,其他coco_2014修改为coco_2017

  5.2 修改DATALOADER配置,注意,针对多卡,以及多机情况时调整该参数能改善数据io性能. 增加节点DATALOADER, 配置为：
      DATALOADER:
        SIZE_DIVISIBILITY: 32
        NUM_WORKERS: 0
6. 执行命令：
  python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_C4_1x.yaml" SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 1
 



