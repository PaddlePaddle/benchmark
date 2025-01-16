# install env
wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}

pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip install ${dir_name}/*

# 安装 deepmd-kit
export DP_ENABLE_TENSORFLOW=0; export USE_TF_PYTHON_LIBS=0 # 禁用tensorflow，防止冲突
pip install -v -e .
pip list
