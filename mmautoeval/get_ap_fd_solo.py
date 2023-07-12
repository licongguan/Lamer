# -*- coding: utf-8 -*
import os
import os.path as osp
from mmautoeval.cal_map import main_map
from cal_fd_solo import main_fd_solo


def write_txt(i, city_1, city_2, sp_label, ap, my_fd_solo):
    my_fd_solo = round(my_fd_solo, 3)
    with open("../utils/ap_and_fd_{}.txt".format(sp_label), "a+") as f:
        f.write('{}-{}-{} AP: {}'.format(i, city_2, sp_label, ap))
        f.write(" || ")
        f.write("{} and {} FD: {}".format(city_1, city_2, my_fd_solo))
        f.write("\n")
    f.close()
    print("{} write success!".format(i))


if __name__ == '__main__':
    # clear txt
    # base_path = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace/'
    # 测试集
    base_path = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace_add/FD-aa/'
    path = ['/media/glc/Elements/DATA/ALSOLO/Public-cityspace/01_AL/FD/aachen/']
    exclude_list = ['1']
    for i in os.listdir(base_path):
        if i not in exclude_list:
            # print("-----------------------------------------------------------", i)
            # exp = base_path + i + "/"
            # map_path = exp + "map/"
            # fd_path = exp + "FD/"
            # for item in os.listdir(fd_path):
            #     path_2 = fd_path + item + "/"
            path_2 = base_path + i + "/"
            city_1 = path[0].split('/')[-2]
            city_2 = path_2.split('/')[-2]
            # cal map
            # my_map, results_flatten = main_map(i.split('_')[0])

            # cal FD_solo
            path.append(path_2)
            spec_label = 0
            labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
            my_fd_solo = main_fd_solo(path, spec_label)

            # print('{}-{}-{} AP: '.format(i, osp.basename(osp.normpath(path_2)), labels[spec_label]),
            #       float(results_flatten[spec_label * 2 + 1]))
            print("{} and {} FD-solo: ".format(osp.basename(osp.normpath(path[0])), osp.basename(osp.normpath(path[1]))),
                  round(my_fd_solo, 3))
            # clear list[1]
            del path[1]
            # write
            # write_txt(i, city_1, city_2, labels[spec_label], results_flatten[spec_label * 2 + 1], my_fd_solo)
