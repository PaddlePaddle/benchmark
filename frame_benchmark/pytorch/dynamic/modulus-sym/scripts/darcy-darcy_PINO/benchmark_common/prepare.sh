# install pytorch
# pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

wget -c --no-proxy ${FLAG_TORCH_WHL_URL}
tar_file_name=$(echo ${FLAG_TORCH_WHL_URL} | awk -F '/' '{print $NF}')
dir_name=$(echo ${tar_file_name} | awk -F '.' '{print $1}')
tar xf ${tar_file_name}
rm -rf ${tar_file_name}
pip install ${dir_name}/*

# install modulus
pushd ../modulus/
pip install -e .
popd

# install modulus-sym
pip install -e .
if [ ! -f './examples_sym.zip' ]; then
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip
fi

if [ ! -d './examples_sym' ]; then
    unzip examples_sym.zip
fi
unalias cp 2>/dev/null
\cp -r -f -v ./examples_sym/examples/* ./examples/

if [ ! -d './examples/darcy/datasets' ]; then
    mkdir -p ./examples/darcy/datasets && cd ./examples/darcy/datasets
    wget https://paddle-qa.bj.bcebos.com/benchmark/pretrained/Darcy_241.tar.gz
    tar xf Darcy_241.tar.gz
    cd -
fi
