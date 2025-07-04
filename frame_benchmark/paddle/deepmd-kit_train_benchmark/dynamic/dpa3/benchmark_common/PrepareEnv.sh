# 安装 deepmd-kit
export DP_ENABLE_TENSORFLOW=0; export USE_TF_PYTHON_LIBS=0 # 禁用tensorflow，防止冲突
pip install -v -e .