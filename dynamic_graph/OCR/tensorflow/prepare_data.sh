# download backbone pretrained model
wget  http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar zxf /resnet_v1_50_2016_08_28.tar.gz

pip install -r requirments.txt
pip install cython
apt-get install libhdf5-dev
pip install h5py
pip install tensorflow-gpu==1.14.0

#download dataset 
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/icdar2015_east.tar
tar xf icdar2015_east.tar


