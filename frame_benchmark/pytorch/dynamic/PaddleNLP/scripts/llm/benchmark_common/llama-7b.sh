mkdir -p huggyllama/llama-7b
cd huggyllama/llama-7b
rm -rf *.json
rm -rf *.safetensors
rm -rf *.model
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/config.json
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/model.safetensors.index.json
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/model-00001-of-00002.safetensors
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/model-00002-of-00002.safetensors
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/tokenizer.model
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/tokenizer_config.json
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/special_tokens_map.json
wget https://paddlenlp.bj.bcebos.com/models/community/huggyllama/llama-7b/torch/generation_config.json