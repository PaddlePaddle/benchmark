echo "*******prepare benchmark start ***********"

# git clone https://github.com/NVIDIA/DeepLearningExamples.git 
cd PyTorch/Translation/Transformer

# download processed data
wget https://paddlenlp.bj.bcebos.com/models/transformers/transformer/wmt14_en_de_joined_dict.tar.gz
mkdir /data
tar -zxvf wmt14_en_de_joined_dict.tar.gz -C /data
rm wmt14_en_de_joined_dict.tar.gz
cp train.py /data
# back to PyTorch/Translation/
cd ..

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com 

pip install --no-cache-dir \
      sacrebleu \
      sentencepiece

apt-get update

apt-get install -y -q cmake pkg-config protobuf-compiler libprotobuf-dev libgoogle-perftools-dev

git clone https://github.com/google/sentencepiece.git sentencepiece

cd  sentencepiece \
  && git checkout d4dd947 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 8 \
  && make install \
  && ldconfig -v

cd ../../Transformer/

export PYTHONPATH=examples/translation/subword-nmt/

ldconfig

git clone https://github.com/rsennrich/subword-nmt.git Transformer/examples/translation/subword-nmt/
git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git checkout ed2ed4d6 && cd ..

pip install -e .
pip install git+https://github.com/NVIDIA/dllogger@v0.1.0#egg=dllogger

echo "*******prepare benchmark end***********"
