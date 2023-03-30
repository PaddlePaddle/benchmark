echo "******prepare benchmark start************"

echo "https_proxy $HTTPS_PRO" 
echo "http_proxy $HTTP_PRO" 
export https_proxy=$HTTPS_PRO
export http_proxy=$HTTP_PRO
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com 

wget ${FLAG_TORCH_WHL_URL}

tar -xf torch_dev_whls.tar

pip install torch_dev_whls/*

pip install transformers pandas psutil scipy

# rm current torch
rm -rf torch_tmp/
mv torch torch_tmp

sed -i '1842,1844d' benchmarks/dynamo/common.py
sed -i '929,933d' benchmarks/dynamo/common.py
sed -i '929i \ \ \ \ \ \ \ \ \ \ \ \ os.environ[\"MASTER_ADDR\"] = os.getenv(\"MASTER_ADDR\", \"localhost\")\n            os.environ[\"MASTER_PORT\"] = os.getenv(\"MASTER_PORT\", \"12355\")\n            os.environ[\"RANK\"] = os.getenv(\"RANK\", \"0\")\n            os.environ[\"WORLD_SIZE\"] = os.getenv(\"WORLD_SIZE\", \"1\")\n            torch.cuda.set_device(int(os.environ[\"RANK\"]))\n            torch.distributed.init_process_group(\n                \"nccl\"\n            )' benchmarks/dynamo/common.py

rm -f ./speedup_eager*
rm -f ./speedup_inductor*

echo "******prepare benchmark end************"
