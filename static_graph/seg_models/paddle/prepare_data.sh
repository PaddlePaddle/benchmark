set -ex

mkdir -p  data/cityscape

# STEP ONE: download data from cityscapes
# register an account on https://www.cityscapes-dataset.com/ and then config the username and password in the below URL
#wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

#STEP TWO: decompress and data and do some preprocess work

unzip gtFine_trainvaltest.zip 
unzip leftImg8bit_trainvaltest.zip

mv gtFine data/cityscape
mv leftImg8bit data/cityscape

git clone https://github.com/mcordts/cityscapesScripts.git
cp -r cityscapesScripts/cityscapesscripts data/cityscape

export PYTHONPATH=$PWD/data/cityscape
python data/cityscape/cityscapesscripts/preparation/createTrainIdLabelImgs.py

wget https://paddle-deeplab.bj.bcebos.com/deeplabv3plus_xception65_initialize.tgz
tar -xf deeplabv3plus_xception65_initialize.tgz && rm deeplabv3plus_xception65_initialize.tgz
