# -*- coding: utf-8 -*
import os

from Config import *
from shutil import copyfile


def get_info():
    # 复制info txt
    for i in os.listdir(imgs_path):
        name = osp.basename(i)
        txt_name = name.split('.')[0] + ".txt"
        try:
            copyfile(osp.join(solo_det_path, "al_info", txt_name), random_AnnCost_path + txt_name)
        except:
            print("{}没有检测到实例".format(i))


def cal_random_AnnCost():
    img_nums = len(os.listdir(imgs_path))
    # 计算该图像注释成本
    img_AnnCost_dict = {}
    all_AnnCost_num = 0
    # 计算一张图片所有实例的点击次数
    for txt in os.listdir(random_AnnCost_path):
        img_AnnCost = cal_AnnCost_img(txt, random_AnnCost_path)
        all_AnnCost_num += img_AnnCost
        # 将每张图像的注释成本加入字典
        img_AnnCost_dict[os.path.join(random_AnnCost_path, txt)] = img_AnnCost
    # 打包 元素1是注释成本 元素2是txt文件名
    my_dict = zip(img_AnnCost_dict.values(), img_AnnCost_dict.keys())
    # 降序排序
    my_dict1 = sorted(my_dict, reverse=True)
    with open(os.path.join(save_random_AnnCost_path), 'w') as f1:
        for i in my_dict1:
            line = str(i).replace("'", '')
            line = line.replace("()", '')
            line = line.strip("()")
            f1.write(line + '\n')
    f1.close()
    print("完成随机选择的样本点击次数排序")

    with open(os.path.join(save_random_all_AnnCost_path), 'w') as f2:
        f2.write("图片个数：{}, 总点击次数：{}".format(img_nums, all_AnnCost_num))
    f2.close()
    print("完成随机选择的样本总点击次数")


def cal_AnnCost_img(txt, txt_path):
    file = open(os.path.join(txt_path, txt), encoding='UTF-8', mode='r')
    img_AnnCost = 0
    for line in file.readlines():
        text = line.strip('\n')
        label = text.split(',')[0]
        confidence = text.split(',')[1]
        cost = int(text.split(',')[2])
        img_AnnCost += cost

    return img_AnnCost


if __name__ == '__main__':
    get_info()
    # 计算点击次数成本
    cal_random_AnnCost()
