# -*- coding: utf-8 -*
# -*- coding: utf-8 -*
from Config import *


def get_ImgNum(save_random_all_AnnCost_path):
    file = open(save_random_all_AnnCost_path, mode='r')
    for line in file.readlines():
        limit_imgnum = int(line.split(", 总点击次数")[0].split("：")[1])

    return limit_imgnum


def get_ours_labeled_txt(limit_imgnum, remain_img_uc_txt_path):
    img_num = 1
    anncosts = 0  # 点击次数累加
    our_labeled_list = []
    file = open(remain_img_uc_txt_path, mode='r')
    for line in file.readlines():
        anncost = int(line.split('],')[0].split(',')[1])
        anncosts += anncost
        # 判断样本数是否超过限制
        if img_num <= limit_imgnum:
            print("图片个数：{}，当前点击次数：{}".format(img_num, anncosts))
            img_anncost_num = "图片个数：{}，当前点击次数：{}".format(img_num, anncosts)
            name = osp.basename(line.split('],')[1]).split('.')[0] + ".png"
            our_labeled_list.append(name)
            img_num += 1  # 计算选择了多少图片了

    # 将ours挑选的样本索引写入到txt
    with open(save_ours_labeled_txt, 'w') as f:
        for i in our_labeled_list:
            line = i + "\n"
            f.write(line)
    f.close()

    # 将ours使用的图片个数和点击次数保存
    with open(save_ours_anncost_txt, 'w') as f:
        line = img_anncost_num
        f.write(line)
    f.close()


if __name__ == '__main__':
    # 获得随机选择的样本总图片个数
    limit_imgnum = get_ImgNum(save_random_all_AnnCost_path)

    # 根据总点击次数获得指定的labeled.txt
    get_ours_labeled_txt(limit_imgnum, remain_img_uc_txt_path)
