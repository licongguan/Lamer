# -*- coding: utf-8 -*
import math
from Config import *
from collections import defaultdict


def save_uc_txt(input_path, class_list):
    class_number = len(class_list)
    img_nums = len(os.listdir(input_path))  # 图像总数

    # 统计每种类别的数量
    count_zero = [0] * class_number
    count_all = num_each_cate(input_path, class_number, class_list, count_zero)
    # print("每种类别的数量按顺序: ", count_all)
    # print("所有类别的数量: ", sum(count_all))

    # 计算每种类别的权重1 以log10为底 参数为该类的所有数量
    weight1_zero = [0] * class_number
    weight1_all = weight1_each_cate(class_number, count_all, weight1_zero)
    # print("权重1: ", weight1_all)

    # 计算一幅图像中每个类别的平均对象数
    mean_num_zero = [0] * class_number
    mean_num_all = mean_num_each_cate(class_number, count_all, mean_num_zero, img_nums)
    # print("一幅图像中每个类别平均对象数: ", mean_num_all)

    # 求和 一幅图像中的对象数量
    sum_num = sum(mean_num_all)
    # print("一幅图像中的对象数量: ", sum_num)

    # 计算每种类别的权重2 (sum_num+C)/(x+1)
    weight2_zero = [0] * class_number
    weight2_all = weight2_each_cate(class_number, weight2_zero, sum_num, mean_num_all)
    # print("权重2: ", weight2_all)

    # 计算该图像不确定性 以及该图像的点击次数
    img_uncertains_dict = {}
    for txt in os.listdir(input_path):
        img_uncertain = cal_uncer_img(txt, input_path, weight1_all, weight2_all, class_list)  # 图像不确定性
        img_AnnCost = cal_AnnCost_img(txt, input_path)  # 图像的点击次数
        # 将不确定性和点击次数加入字典
        img_uncertains_dict[os.path.join(input_path, txt)] = [img_uncertain, img_AnnCost]

    # 打包 按照不确定性排序
    my_dict = zip(img_uncertains_dict.values(), img_uncertains_dict.keys())
    my_dict1 = sorted(my_dict, reverse=True)   # 降序排序

    # 保存
    with open(os.path.join(remain_img_uc_txt_path), 'w') as f1:
        for i in my_dict1:
            line = str(i).replace("'", '')
            line = line.replace("()", '')
            line = line.strip("()")
            f1.write(line + '\n')
    f1.close()
    print("ours_剩余未标注样本的不确定性得分统计")


def num_each_cate(txt_path, class_number, class_all, count_zero):
    for txt in os.listdir(txt_path):
        file = open(os.path.join(txt_path, txt), encoding='UTF-8', mode='r')
        for line in file.readlines():
            text = line.strip('\n')
            for i in range(class_number):
                if text.split(',')[0] in class_all[i]:
                    count_zero[i] += 1

    return count_zero


def weight1_each_cate(class_number, count_all, weight1_zero):
    for i in range(class_number):
        weight1_zero[i] = math.log10(count_all[i])

    return weight1_zero


def mean_num_each_cate(class_number, count_all, mean_num_zero, img_nums):
    for i in range(class_number):
        mean_num_zero[i] = float(count_all[i]) / img_nums

    return mean_num_zero


def weight2_each_cate(class_number, weight2_zero, sum_num, mean_num_all):
    for i in range(class_number):
        weight2_zero[i] = (sum_num + class_number) / (mean_num_all[i] + 1)

    return weight2_zero


def cal_uncer_img(txt, txt_path, weight1_all, weight2_all, class_all):
    file = open(os.path.join(txt_path, txt), encoding='UTF-8', mode='r')
    img_uncertain = 0
    for line in file.readlines():
        text = line.strip('\n')
        label = text.split(',')[0]
        confidence = text.split(',')[1]
        w1 = weight1_all[class_all.index(label)]
        w2 = weight2_all[class_all.index(label)]
        single_obj = w1 * w2 * (1 - float(confidence))
        img_uncertain += single_obj
    return img_uncertain


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
    save_uc_txt(osp.join(solo_det_path + "al_info/"), CLASSES)
