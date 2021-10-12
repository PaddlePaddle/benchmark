#if [ -f Miniconda3-latest-Linux-x86_64.sh];then
#    wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#    bash Miniconda3-latest-Linux-x86_64.sh
#fi
#
#. /workspace/miniconda3/bin/activate

#conda create -n wenet python=3.8
#conda activate wenet


test -d venv || virtualenv -p python3.7 venv
source venv/bin/activate

pushd models/wenet

pip install -r requirements.txt
#conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

popd

# 替换wenet原始项目当中的executor.py
cp scripts/executor.py models/wenet/wenet/utils/



#apt-key adv --keyserver keyserver.ubuntu.com --recv-keys CC86BB64
#apt-get install software-properties-common -y
#add-apt-repository ppa:rmescandon/yq
#apt update
#apt install yq -y


VERSION=v4.2.0
BINARY=yq_linux_amd64
wget https://github.com/mikefarah/yq/releases/download/${VERSION}/${BINARY} -O /usr/bin/yq &&\
        chmod +x /usr/bin/yq
