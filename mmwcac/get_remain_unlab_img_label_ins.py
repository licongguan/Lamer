# -*- coding: utf-8 -*
# 按照半监督学习的U2PL设置的挑选样本
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
        print("正在拷贝指定的第{}张图片: {}".format(image_count, name))
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
        # annotation_name = image_name + '_instanceIds.png'
        # 读instanceIds.png 单通道图片 array 1024 2048
        annotation = cv2.imread(os.path.join(spec_train_gtFine_path, annotation_name), -1)
        h, w = annotation.shape[:2]
        #  np.unique(annotation) 计算 numpy 数组中每个唯一元素的出现次数，我们可以使用 numpy.unique() 函数 它将数组作为输入参数，并按升序返回数组内的所有唯一元素 label的个数
        #  1,2,3,4,7,11,17,19,21,22,23,24,24000,24001,24002,24003,24004,24005,24006,26000,26001,26002,26003,26004,26005,26006,26007,26008,26009,26010,26011,26012,26013,28000,28001,28002
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
            # 把实例的地方变成255 其他是0像素 二值图
            instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = annotation == id
            instance_mask[mask] = 255
            mask_name = image_name + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(ann_folder, mask_name), instance_mask)
            idx += 1
        pic_count += 1
        print("正在生成指定的第{}张图片的mask_ins: {}".format(pic_count, pic_name))


if __name__ == '__main__':
    # 实例分割排除的类别序号
    background_label = list(range(-1, 24, 1)) + [29, 30, 34]

    # 提取指定城市的img和gtFine到一个文件夹all_img_path all_gtFine_path
    get_spec_city_to_single_dir(all_img_path, all_gtFine_path, unlabeled_txt, unlab_img_path, unlab_gtFine_path)
    print("---------------------------------------------------------------图片提取完成---------------------------------------------------------------")

    # 处理每一个gtFine_instanceids.png，每个png对应一个文件夹，存放该图片的所有实例mask
    get_gtFinePNG_ins_mask(unlab_img_path, unlab_gtFine_path, unlab_instance_path)
    print("---------------------------------------------------------------实例mask提取完成---------------------------------------------------------------")


