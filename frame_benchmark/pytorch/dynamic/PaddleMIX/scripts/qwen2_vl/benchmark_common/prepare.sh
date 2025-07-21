rm -rf /root/.paddlemix/datasets/playground
rm -rf /root/.paddlemix/torch_models

# dataset
mkdir -p /root/.paddlemix/datasets
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar
tar -xf playground.tar
mv playground /root/.paddlemix/datasets/
rm -rf playground.tar
ln -s /root/.paddlemix/datasets/playground data/

# pretrain model from hf
mkdir -p /root/.paddlemix/torch_models
wget https://paddlenlp.bj.bcebos.com/datasets/benchmark/torch_models/Qwen2-VL-2B-Instruct.tar
wget https://paddlenlp.bj.bcebos.com/datasets/benchmark/torch_models/Qwen2-VL-7B-Instruct.tar
tar -xf Qwen2-VL-2B-Instruct.tar
tar -xf Qwen2-VL-7B-Instruct.tar
mv Qwen2-VL-2B-Instruct /root/.paddlemix/torch_models
mv Qwen2-VL-7B-Instruct /root/.paddlemix/torch_models
rm -f Qwen2-VL-2B-Instruct.tar
rm -f Qwen2-VL-7B-Instruct.tar
ln -s /root/.paddlemix/torch_models/Qwen2-VL-2B-Instruct ./
ln -s /root/.paddlemix/torch_models/Qwen2-VL-7B-Instruct ./

pip install liger-kernel

# export http_proxy=agent.baidu.com:8188
# export https_proxy=agent.baidu.com:8188

# env
pip install -r requirements.txt
