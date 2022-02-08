echo "*******prepare benchmark start ***********"

# git clone https://github.com/NVIDIA/DeepLearningExamples.git 
cd PyTorch/Translation/Transformer
chmod 777 scripts/*
scripts/run_preprocessing.sh
# back to PyTorch/Translation/
cd ..


pip install --no-cache-dir \
      sacrebleu \
      sentencepiece

pip install fairseq

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
cd ..
git clone https://github.com/rsennrich/subword-nmt.git Transformer/examples/translation/subword-nmt/
git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git checkout ed2ed4d6 && cd ..

cd Transformer
pip install -e .
pip install git+https://github.com/NVIDIA/dllogger@v0.1.0#egg=dllogger

echo "*******prepare benchmark end***********"
