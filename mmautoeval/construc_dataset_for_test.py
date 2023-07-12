# -*- coding: utf-8 -*
"""
1.FD/xxx_foggy_0.01/
2.map/coco/test2017 map/coco/annotations_test2017
3./media/glc/Elements/project/SOLO/MyTrain/ALSOLO/Public-cityspace/018_AL/solov2_Public_cityspace_018_AL.py
"""
import os
import os.path as osp
from shutil import copyfile


def get_FD_city_foggy(beta):
    i = 37
    for city in os.listdir(foggy_img_path):
        save_beta_img_path = save_map_FD_path + '0{}_AL/FD/{}_foggy_{}/'.format(i, city, beta)
        if not osp.exists(save_beta_img_path):
            os.makedirs(save_beta_img_path)
        # aachen
        for file in os.listdir(foggy_img_path + city):
            copyfile(osp.join(foggy_img_path + city, file), save_beta_img_path + file)
            print(file + " success!")
        i += 1


def get_map_city_foggy():
    i = 37
    for city in os.listdir(foggy_img_path):
        target_img_path = save_map_FD_path + '0{}_AL/map/coco/test2017/'.format(i)
        target_json_path = save_map_FD_path + '0{}_AL/map/coco/annotations_test2017/'.format(i)
        if not osp.exists(target_img_path):
            os.makedirs(target_img_path)
        if not osp.exists(target_json_path):
            os.makedirs(target_json_path)

        # aachen
        for file in os.listdir(foggy_img_path + city):
            copyfile(osp.join(foggy_img_path + city, file), target_img_path + file)
            copyfile(cityspace_json_all + file.replace(".jpg", ".json"), target_json_path + file.replace(".jpg", ".json"))
            print(file + " success!")
        i += 1


def get_config():
    i = 37
    for city in os.listdir(foggy_img_path):
        target_config_path = save_config_path + '0{}_AL/'.format(i)
        if not osp.exists(target_config_path):
            os.makedirs(target_config_path)
        config_name = 'solov2_Public_cityspace_0{}_AL.py'.format(i)
        copyfile(default_config_path, target_config_path + config_name)

        # correct config  67hang
        with open(target_config_path + config_name, 'r') as f:
            lines = f.readlines()
            lines[66] = "data_root = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace_test_FD_AP/0{}_AL/map/coco/'".format(i) + '\n'
        with open(target_config_path + config_name, 'w') as f:
            for data in lines:
                f.write(data)
        i += 1


if __name__ == '__main__':
    foggy_img_path = '/media/glc/Elements/DATA/Public data set/CitySpace/MyCitySpace/ImgFoggy/foggy_beta_0.02/'
    cityspace_json_all = '/media/glc/Elements/DATA/Public data set/CitySpace/MyCitySpace/labelme8类注/annotations_train2017/'
    save_map_FD_path = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace_test_FD_AP/'
    if not osp.exists(save_map_FD_path):
        os.makedirs(save_map_FD_path)
    default_config_path = '/media/glc/Elements/project/SOLO/MyTrain/ALSOLO/Public-cityspace/02_AL/solov2_Public_cityspace_02_AL.py'
    save_config_path = '/media/glc/Elements/project/SOLO/MyTrain/ALSOLO/Public-cityspace/'
    if not osp.exists(save_config_path):
        os.makedirs(save_config_path)
    beta = 0.02
    # FD/xxx_foggy_0.01/
    get_FD_city_foggy(beta)
    print("FD Dataset success!")

    # map/coco/test2017 map/coco/annotations_test2017
    get_map_city_foggy()
    print("map Dataset success!")

    # /media/glc/Elements/project/SOLO/MyTrain/ALSOLO/Public-cityspace/018_AL/solov2_Public_cityspace_018_AL.py
    get_config()
    print("config success!")


