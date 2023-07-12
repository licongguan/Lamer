# -*- coding: utf-8 -*
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile
import random
from Config import *

random.seed(42)


def main():
    un_lab_img_list = glob.glob(unlab_img_path + "/*.png")

    _, label_list = train_test_split(un_lab_img_list, test_size=proportion_num, random_state=42)
    print("Number of annotations:", len(label_list))

    assert len(label_list) == label_num

    get_imgs(label_list, imgs_path)
    get_gtFines(label_list, gtFines_path)
    get_instances(imgs_path, gtFines_path, instances_path)
    get_labeled_txt(label_list)


def get_imgs(label_list, imgs_path):
    for i in label_list:
        name = osp.basename(i)
        copyfile(unlab_img_path + name, imgs_path + name)
        print("{} img success".format(name))


def get_gtFines(label_list, gtFines_path):
    for i in label_list:
        name = osp.basename(i)
        ann_name = name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_gtFine_instanceIds.png'
        copyfile(unlab_gtFine_path + ann_name, gtFines_path + ann_name)
        print("{} gtFine success".format(name))


def get_instances(imgs_path, gtFines_path, instances_path):
    global idx
    idx = 0
    pic_count = 0
    for pic_name in os.listdir(imgs_path):
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(instances_path, image_name)
        if not os.path.exists(ann_folder):
            os.mkdir(ann_folder)
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'

        annotation = cv2.imread(os.path.join(gtFines_path, annotation_name), -1)
        h, w = annotation.shape[:2]
        ids = np.unique(annotation)
        for id in ids:
            if id in background_label:
                continue
            else:
                class_id = id // 1000
                if class_id == 24:
                    instance_class = 'person'
                elif class_id == 25:
                    instance_class = 'rider'
                elif class_id == 26:
                    instance_class = 'car'
                elif class_id == 27:
                    instance_class = 'truck'
                elif class_id == 28:
                    instance_class = 'bus'
                elif class_id == 31:
                    instance_class = 'train'
                elif class_id == 32:
                    instance_class = 'motorcycle'
                elif class_id == 33:
                    instance_class = 'bicycle'
                else:
                    continue
            instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255
            mask_name = image_name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(ann_folder, mask_name), instance_mask)
            idx += 1
        pic_count += 1
        print("{}: {} instance success".format(pic_count, pic_name))


def get_labeled_txt(label_list):
    with open(save_random_labeled_txt, 'w') as f:
        for i in label_list:
            line = osp.basename(i) + "\n"
            f.write(line)
    f.close()


if __name__ == '__main__':
    main()
