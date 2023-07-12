# -*- coding: utf-8 -*
from Config import *
from shutil import copyfile
import cv2
import numpy as np


def main():
    label_list = []
    file = open(save_ours_labeled_txt, mode='r')
    for line in file.readlines():
        line = line.strip("\n")
        label_list.append(line)

    print("标注的数量:", len(label_list))

    ours_anncost_txt = open(save_ours_anncost_txt, mode='r')
    for line in ours_anncost_txt.readlines():
        label_num = line.split("，")[0].split("：")[1]

    assert len(label_list) == int(label_num)  # 判断ours_labeled.txt里面的图片索引个数和ours_anncost.txt的个数一致

    # 提取img
    get_imgs(label_list, ours_imgs_path)
    # 提取gtFine
    get_gtFines(label_list, ours_gtFines_path)
    # 提取instance
    get_instances(ours_imgs_path, ours_gtFines_path, ours_instances_path)


def get_imgs(label_list, ours_imgs_path):
    for i in label_list:
        name = osp.basename(i)
        copyfile(unlab_img_path + name, ours_imgs_path + name)
        print("{}的img success".format(name))


def get_gtFines(label_list, ours_gtFines_path):
    for i in label_list:
        name = osp.basename(i)
        ann_name = name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_gtFine_instanceIds.png'
        copyfile(unlab_gtFine_path + ann_name, ours_gtFines_path + ann_name)
        print("{}的gtFine success".format(name))


def get_instances(ours_imgs_path, ours_gtFines_path, ours_instances_path):
    global idx
    idx = 0
    pic_count = 0
    for pic_name in os.listdir(ours_imgs_path):
        image_name = pic_name.split('.')[0]
        ann_folder = os.path.join(ours_instances_path, image_name)
        if not os.path.exists(ann_folder):
            os.mkdir(ann_folder)
        annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[2] + '_gtFine_instanceIds.png'
        # annotation_name = image_name + '_instanceIds.png'
        # 读instanceIds.png 单通道图片 array 1024 2048
        annotation = cv2.imread(os.path.join(ours_gtFines_path, annotation_name), -1)
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
        print("{}: {}的instance success".format(pic_count, pic_name))


if __name__ == '__main__':
    main()
