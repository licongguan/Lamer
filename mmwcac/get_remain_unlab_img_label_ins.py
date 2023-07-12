# -*- coding: utf-8 -*
import cv2
import json
from pycococreatortools import *
from shutil import copyfile
from Config import *


def get_spec_city_to_single_dir(all_img_path, all_gtFine_path, specified_city_txt, spec_train_images_path, spec_train_gtFine_path):
    image_count = 1
    file = open(specified_city_txt, encoding='UTF-8', mode='r')
    for line in file.readlines():
        img_name = line.strip('\n')
        name = img_name.split('/')[3]
        ann_name = name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_gtFine_instanceIds.png'
        copyfile(os.path.join(all_img_path + name), os.path.join(spec_train_images_path + name))
        copyfile(os.path.join(all_gtFine_path + ann_name), os.path.join(spec_train_gtFine_path + ann_name))
        print("Copying the {} th image specified: {}".format(image_count, name))
        image_count += 1


def get_gtFinePNG_ins_mask(spec_train_images_path, spec_train_gtFine_path, spec_train_instance_dir):
    global idx
    idx = 0
    pic_count = 0
    for pic_name in os.listdir(spec_train_images_path):
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(spec_train_instance_dir, image_name)
        if not os.path.exists(ann_folder):
            os.mkdir(ann_folder)
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'
        annotation = cv2.imread(os.path.join(spec_train_gtFine_path, annotation_name), -1)
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
        print("Generating mask ins for the {} th image specified: {}".format(pic_count, pic_name))


if __name__ == '__main__':
    background_label = list(range(-1, 24, 1)) + [29, 30, 34]

    get_spec_city_to_single_dir(all_img_path, all_gtFine_path, unlabeled_txt, unlab_img_path, unlab_gtFine_path)

    # Process each gt Fine instanceids.png, with each png corresponding to a folder that stores all instance masks of the image
    get_gtFinePNG_ins_mask(unlab_img_path, unlab_gtFine_path, unlab_instance_path)


