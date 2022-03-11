echo "*******prepare benchmark start ***********"


cd PyTorch/LanguageModeling/BERT

ln -s  $PWD /workspace/bert

mkdir /workspace/bert/data

export http_proxy=http://172.19.57.45:3128
export https_proxy=http://172.19.57.45:3128
export ftp_proxy=http://172.19.57.45:3128/
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

export BERT_PREP_WORKING_DIR=/workspace/bert/data

bash data/create_datasets_from_start.sh



cd /workspace/bert

pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar onnxruntime tokenizers>=0.7\
 git+https://github.com/NVIDIA/dllogger wget

apt-get install -y iputils-ping

# Install lddl
conda install -y jemalloc
pip install /workspace/bert/lddl
python -m nltk.downloader punkt

RUN pip install lamb_amp_opt/

echo "*******prepare benchmark end***********"
