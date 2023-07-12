# -*- coding: utf-8 -*
from Config import *


def get_ImgNum(save_random_all_AnnCost_path):
    file = open(save_random_all_AnnCost_path, mode='r')
    for line in file.readlines():
        limit_imgnum = int(line.split(", Total clicks")[0].split("：")[1])

    return limit_imgnum


def get_ours_labeled_txt(limit_imgnum, remain_img_uc_txt_path):
    img_num = 1
    anncosts = 0  # 点击次数累加
    our_labeled_list = []
    file = open(remain_img_uc_txt_path, mode='r')
    for line in file.readlines():
        anncost = int(line.split('],')[0].split(',')[1])
        anncosts += anncost
        if img_num <= limit_imgnum:
            print("Number of images：{}，Current clicks：{}".format(img_num, anncosts))
            img_anncost_num = "Number of images：{}，Current clicks：{}".format(img_num, anncosts)
            name = osp.basename(line.split('],')[1]).split('.')[0] + ".png"
            our_labeled_list.append(name)
            img_num += 1

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
    # Obtain the total number of randomly selected sample images
    limit_imgnum = get_ImgNum(save_random_all_AnnCost_path)

    # Obtain the specified labeled.txt based on the total number of clicks
    get_ours_labeled_txt(limit_imgnum, remain_img_uc_txt_path)
