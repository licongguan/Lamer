# -*- coding: utf-8 -*
import os
import math
import os.path as osp


CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
base_dir = '../data/public_dataset/CITYSCAPES/cityscapes2coco-master/cityscapes/'
mode = 'train'

all_img_path = osp.join(base_dir, "cityscapes_8instance_to_coco_json/{}_images/".format(mode))
all_gtFine_path = osp.join(base_dir, "cityscapes_8instance_to_coco_json/{}_gtFine/".format(mode))

style = '372'  # 组别设置  186  372  744  1488
remain = str(2975 - int(style))  # 2789 2603 2231 1487

exp_path = osp.join(base_dir, "data_splits/cityscapes/")

unlabeled_txt = osp.join(exp_path, style, "suponly", "unlabeled.txt")  # remain剩余的未标注数据索引
unlab_img_path = osp.join(exp_path, style, "suponly", remain, 'unlabeled_images/')
unlab_gtFine_path = osp.join(exp_path, style, "suponly", remain, 'unlabeled_gtFine/')
unlab_instance_path = osp.join(exp_path, style, "suponly", remain, 'unlabeled_instance_dir/')
if not osp.exists(unlab_img_path):
    os.makedirs(unlab_img_path)
    os.makedirs(unlab_gtFine_path)
    os.makedirs(unlab_instance_path)

proportion_num = 0.25
label_num = math.ceil(int(remain) * proportion_num)
background_label = list(range(-1, 24, 1)) + [29, 30, 34]

imgs_path = osp.join(exp_path, style, "random", str(label_num), "images/")
gtFines_path = osp.join(exp_path, style, "random", str(label_num), "gtFines/")
instances_path = osp.join(exp_path, style, "random", str(label_num), "instances/")
save_random_labeled_txt = osp.join(exp_path, style, "random", str(label_num), "random_labeled.txt")
if not osp.exists(imgs_path):
    os.makedirs(imgs_path)
    os.makedirs(gtFines_path)
    os.makedirs(instances_path)

# 将新标注的加入到源数据集 生成coco
init_labeled_img = osp.join(exp_path, style, "suponly", 'spec_train_images/')
init_labeled_instance = osp.join(exp_path, style, "suponly", 'spec_train_instance_dir/')

random_coco_path = osp.join(exp_path, style, "random", str(label_num), 'coco/')
random_train2017_path = osp.join(random_coco_path, 'train2017/')
random_instance2017_path = osp.join(random_coco_path, 'instance2017/')
random_annotations_path = osp.join(random_coco_path, 'annotations/')
if not osp.exists(random_coco_path):
    os.makedirs(random_coco_path)
    os.makedirs(random_train2017_path)
    os.makedirs(random_instance2017_path)
    os.makedirs(random_annotations_path)

# solo检测结果保存
solo_det_path = osp.join(exp_path, style, "suponly", remain, 'solo_det_result/')
if not os.path.exists(solo_det_path):
    os.makedirs(solo_det_path)
    os.makedirs(solo_det_path + "whole")
    os.makedirs(solo_det_path + "single")
    os.makedirs(solo_det_path + "json")
    os.makedirs(solo_det_path + "al_info/")

# 检测参数相关
config_file = '../MyTrain/ALSOLO/experiments/cityscapes/{}/suponly/solov2_light_512_dcn_r50_fpn_8gpu_3x.py'.format(style)
checkpoint_file = '../MyTrain/ALSOLO/experiments/cityscapes/{}/suponly/epoch_144.pth'.format(style)
det_thr = 0.25  # 检测阈值
class_name_to_id = {"_background_": 0, 'person': 1, 'rider': 2, 'car': 3, 'truck': 4, 'bus': 5, 'train': 6, 'motorcycle': 7, 'bicycle': 8, }
sort_by_density = True
every_ins_viz = True
generate_json = True

# 计算随机挑选的样本的计算成本
random_AnnCost_path = osp.join(exp_path, style, "random", str(label_num), "AnnCost/")
if not os.path.exists(random_AnnCost_path):
    os.makedirs(random_AnnCost_path)
save_random_AnnCost_path = osp.join(exp_path, style, "random", str(label_num), "Sort randomly selected samples by number of clicks.txt")
save_random_all_AnnCost_path = osp.join(exp_path, style, "random", str(label_num), "The total number of clicks on randomly selected samples.txt")

# ours 样本选择
ours_path = osp.join(exp_path, style, "ours", str(label_num))
if not os.path.exists(ours_path):
    os.makedirs(ours_path)
remain_img_uc_txt_path = osp.join(ours_path, "ours_Uncertainty score of remaining unlabeled samples.txt")
save_ours_labeled_txt = osp.join(ours_path, "ours_labeled.txt")
save_ours_anncost_txt = osp.join(ours_path, "ours_anncost.txt")

# ours标注
ours_imgs_path = osp.join(ours_path, "images/")
ours_gtFines_path = osp.join(ours_path, "gtFines/")
ours_instances_path = osp.join(ours_path, "instances/")
if not osp.exists(ours_imgs_path):
    os.makedirs(ours_imgs_path)
    os.makedirs(ours_gtFines_path)
    os.makedirs(ours_instances_path)

# 将ours新标注的加入到源数据集 生成coco
ours_coco_path = osp.join(ours_path, 'coco/')
ours_train2017_path = osp.join(ours_coco_path, 'train2017/')
ours_instance2017_path = osp.join(ours_coco_path, 'instance2017/')
ours_annotations_path = osp.join(ours_coco_path, 'annotations/')
if not osp.exists(ours_coco_path):
    os.makedirs(ours_coco_path)
    os.makedirs(ours_train2017_path)
    os.makedirs(ours_instance2017_path)
    os.makedirs(ours_annotations_path)