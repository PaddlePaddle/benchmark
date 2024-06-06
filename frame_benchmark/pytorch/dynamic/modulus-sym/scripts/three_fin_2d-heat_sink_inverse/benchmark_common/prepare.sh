
# install modulus
pushd ../modulus/
pip install -e .
popd

# install modulus-sym
pip install -e .

# install pytorch
pip install https://paddle-qa.bj.bcebos.com/benchmark/pretrained/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl

if [ ! -f './examples_sym.zip' ]; then
    wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip
fi

if [ ! -d './examples_sym' ]; then
    unzip examples_sym.zip
fi
unalias cp 2>/dev/null
\cp -r -f -v ./examples_sym/examples/* ./examples/
