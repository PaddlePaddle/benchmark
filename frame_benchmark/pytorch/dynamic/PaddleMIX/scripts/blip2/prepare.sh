
rm -rf /export/home/.cache/lavis/coco/
# dataset
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/blip2/coco.tar.gz
tar -zxvf coco.tar.gz
mv coco /export/home/.cache/lavis/
rm -rf coco
rm -rf coco.tar.gz
# env
pip install -r scripts/blip2/requirements.txt
