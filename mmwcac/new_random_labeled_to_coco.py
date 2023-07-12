# -*- coding: utf-8 -*
import json
from pycococreatortools import *
from Config import *
from shutil import copyfile, copytree


def get_coco_json(train2017_path, instance2017_path, annotations_path):
    person = 0
    rider = 0
    car = 0
    truck = 0
    bus = 0
    train = 0
    motorcycle = 0
    bicycle = 0

    files = os.listdir(train2017_path)

    INFO, LICENSES, CATEGORIES = coco_table()

    coco_output = {

        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # go through each image
    for image_filename in files:
        image_name = image_filename.split('.')[0]
        image_path = os.path.join(train2017_path, image_filename)
        image = Image.open(image_path)
        # 调用外部函数
        image_info = create_image_info(image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        annotation_sub_path = os.path.join(instance2017_path, image_name)
        ann_files = os.listdir(annotation_sub_path)
        if len(ann_files) == 0:
            print("no avaliable annotation")
            continue
        else:
            for annotation_filename in ann_files:
                annotation_path = os.path.join(annotation_sub_path, annotation_filename)
                for x in CATEGORIES:
                    if x['name'] in annotation_filename:
                        class_id = x['id']
                        break
                if class_id == 1:
                    person += 1
                elif class_id == 2:
                    rider += 1
                elif class_id == 3:
                    car += 1
                elif class_id == 4:
                    truck += 1
                elif class_id == 5:
                    bus += 1
                elif class_id == 6:
                    train += 1
                elif class_id == 7:
                    motorcycle += 1
                elif class_id == 8:
                    bicycle += 1
                else:
                    print('illegal class id')
                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                # 获取mask轮廓
                binary_mask = np.asarray(Image.open(annotation_path).convert('1')).astype(np.uint8)

                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
            print("{}: {}的points success".format(image_id, image_filename))

    with open(osp.join(annotations_path, "instances_train2017.json"), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    print("各类别数量:")
    print("person: {}, rider: {}, car: {}, truck: {}, bus: {}, train: {}, motorcycle: {}, bicycle: {}".format(person, rider, car, truck, bus, train, motorcycle, bicycle))

    with open(osp.join(random_coco_path, "train_num.txt"), 'w') as f:
        line = "person: {}, rider: {}, car: {}, truck: {}, bus: {}, train: {}, motorcycle: {}, bicycle: {}".format(person, rider, car, truck, bus, train, motorcycle, bicycle)
        f.write(line + '\n')
    f.close()


def coco_table():
    INFO = {
        "description": "Cityscapes_Instance Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": "2020",
        "contributor": "Kevin_Jia",
        "date_created": "2020-1-23 19:19:19.123456"
    }

    LICENSES = [
        {

            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {

            'id': 1,
            'name': 'person',
            'supercategory': 'cityscapes',
        },
        {

            'id': 2,
            'name': 'rider',
            'supercategory': 'cityscapes',
        },
        {

            'id': 3,
            'name': 'car',
            'supercategory': 'cityscapes',
        },
        {

            'id': 4,
            'name': 'truck',
            'supercategory': 'cityscapes',
        },
        {

            'id': 5,
            'name': 'bus',
            'supercategory': 'cityscapes',
        },
        {

            'id': 6,
            'name': 'train',
            'supercategory': 'cityscapes',
        },
        {

            'id': 7,
            'name': 'motorcycle',
            'supercategory': 'cityscapes',
        },
        {

            'id': 8,
            'name': 'bicycle',
            'supercategory': 'cityscapes',
        }
    ]

    return INFO, LICENSES, CATEGORIES


if __name__ == '__main__':
    # 把两次标注的数据放到一起
    for i in os.listdir(init_labeled_img):  # init imgs
        copyfile(init_labeled_img + i, random_train2017_path + i)
    for i in os.listdir(imgs_path):  # labeled imgs
        copyfile(imgs_path + i, random_train2017_path + i)

    for i in os.listdir(init_labeled_instance):  # int instances
        copytree(init_labeled_instance + i, random_instance2017_path + i)
    for i in os.listdir(instances_path):  # labeled instances
        copytree(instances_path + i, random_instance2017_path + i)

    # 生成coco json
    get_coco_json(random_train2017_path, random_instance2017_path, random_annotations_path)
