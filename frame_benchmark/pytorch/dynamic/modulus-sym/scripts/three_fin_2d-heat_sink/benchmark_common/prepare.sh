# install pytorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# install modulus
pushd ../modulus/
pip install -e .
popd

# install modulus-sym
pip install -e .
