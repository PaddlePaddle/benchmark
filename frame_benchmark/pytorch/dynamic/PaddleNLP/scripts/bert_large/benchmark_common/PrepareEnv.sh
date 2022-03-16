echo "*******prepare benchmark start ***********"


cd PyTorch/LanguageModeling/BERT

ln -s  $PWD /workspace/bert

mkdir /workspace/bert/data

export BERT_PREP_WORKING_DIR=/workspace/bert/data


apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

cd /workspace
git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 6408a430fc504a38b04d37ce5e7fc740191dee16 && cd ..
git clone https://github.com/soskek/bookcorpus.git

apt-get install libb64-0d

cd /workspace/bert
pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar onnxruntime \
 git+https://github.com/NVIDIA/dllogger wget

apt-get install -y iputils-ping

#download data, extract data
# bash data/create_datasets_from_start.sh
cd /workspace/bert/data/
# hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en
wget https://bj.bcebos.com/paddlenlp/data/bert_512_wiki_demo_data.zip
unzip bert_512_wiki_demo_data.zip

echo "*******prepare benchmark end***********"
