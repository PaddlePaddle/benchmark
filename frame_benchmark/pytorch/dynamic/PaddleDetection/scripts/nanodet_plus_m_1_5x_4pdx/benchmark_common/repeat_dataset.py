import argparse
import json
import os
import shutil


def repeat_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', default='1', help="repeat nums")
    parser.add_argument(
        '--file_dir', default='', help="the dir of coco dataset json file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = repeat_args()
    assert os.path.exists(os.path.join(args.file_dir, 'instance_train.json')
                          ), "Please prepare the dataset first"
    if os.path.exists(os.path.join(args.file_dir, 'instance_train_old.json')):
        input_file = os.path.join(args.file_dir, 'instance_train_old.json')
    else:
        shutil.copy(
            os.path.join(args.file_dir, 'instance_train.json'),
            os.path.join(args.file_dir, 'instance_train_old.json'))
        input_file = os.path.join(args.file_dir, 'instance_train_old.json')
    # 读取COCO数据集的JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    images = []
    annotations = []
    categories = []
    annotation_id_nums = 0

    # 提取images字段
    image_id_nums = len(data['images'])
    for annotation in data['annotations']:
        annotation_id_nums = max(annotation['id'], annotation_id_nums)
    for image in data['images']:
        license = image['license']
        coco_url = image['coco_url']
        date_captured = image['date_captured']
        flickr_url = image['flickr_url']
        file_name = image['file_name']
        height = image['height']
        width = image['width']
        image_id = image['id']
        for i in range(int(args.repeat)):
            image_dict = {
                'license': license,
                'file_name': file_name,
                'coco_url': coco_url,
                'height': height,
                'width': width,
                'date_captured': date_captured,
                'flickr_url': flickr_url,
                'id': image_id_nums * i + image_id
            }
            images.append(image_dict)

    # 提取annotations字段
    for annotation in data['annotations']:
        annotation_id = annotation['id']
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        iscrowd = annotation['iscrowd']
        area = annotation['area']
        segmentation = annotation['segmentation']
        for j in range(int(args.repeat)):
            annotation_dict = {
                'area': area,
                'id': annotation_id_nums * j + annotation_id,
                'image_id': image_id_nums * j + image_id,
                'category_id': category_id,
                'bbox': bbox,
                'iscrowd': iscrowd,
                'segmentation': segmentation
            }
            annotations.append(annotation_dict)

    # 提取categories字段
    for category in data['categories']:
        category_id = category['id']
        name = category['name']
        categories.append({'id': category_id, 'name': name})

    # 将数据转换为标准coco格式的字典对象
    coco_dataset = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(os.path.join(args.file_dir, 'instance_train.json'), 'w') as f:
        json.dump(coco_dataset, f)
