# -*- coding: utf-8 -*
from Config import *


def get_AnnCost(save_random_all_AnnCost_path):
    file = open(save_random_all_AnnCost_path, mode='r')
    for line in file.readlines():
        limit_anncost = int(line.split("Total clicks：")[1])

    return limit_anncost


def get_ours_labeled_txt(limit_anncost, remain_img_uc_txt_path):
    img_num = 1
    anncosts = 0  # Accumulated click count
    our_labeled_list = []
    file = open(remain_img_uc_txt_path, mode='r')
    for line in file.readlines():
        anncost = int(line.split('],')[0].split(',')[1])
        anncosts += anncost
        # Determine if the total number of clicks exceeds the limit
        if anncosts <= limit_anncost:
            print("Number of images：{}，Current clicks：{}".format(img_num, anncosts))
            img_anncost_num = "Number of images：{}，Current clicks：{}".format(img_num, anncosts)
            name = osp.basename(line.split('],')[1]).split('.')[0] + ".png"
            our_labeled_list.append(name)
            img_num += 1  # 计算选择了多少图片了

    # Write the sample index selected by ours to txt
    with open(save_ours_labeled_txt, 'w') as f:
        for i in our_labeled_list:
            line = i + "\n"
            f.write(line)
    f.close()

    # Save the number of images and clicks used by ours
    with open(save_ours_anncost_txt, 'w') as f:
        line = img_anncost_num
        f.write(line)
    f.close()


if __name__ == '__main__':
    # Obtain the total number of clicks for randomly selected samples
    limit_anncost = get_AnnCost(save_random_all_AnnCost_path)

    # Obtain the specified labeled.txt based on the total number of clicks
    get_ours_labeled_txt(limit_anncost, remain_img_uc_txt_path)
