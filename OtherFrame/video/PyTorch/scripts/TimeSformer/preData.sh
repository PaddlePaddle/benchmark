################################# 准备训练数据
# 完成，存放至 TimeSformer/videos下
wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" # 下载训练数据
unrar x UCF101.rar # 解压
mv ./UCF-101 ./videos # 重命名文件夹为./videos
rm -rf ./UCF101.rar
