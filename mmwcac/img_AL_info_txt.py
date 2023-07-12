# -*- coding: utf-8 -*
from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import PIL.Image
import imgviz
from shutil import copyfile
from pycococreatortools import *
import base64
import json
from Config import *


class Network:
    def __init__(self):
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def segmentation(self, image_path):
        results = inference_detector(self.model, image_path)
        return results


def main(i, result, class_name_to_id, score_thr, sort_by_density=True):
    img_pil = PIL.Image.open(unlab_img_path + i)
    img = np.array(img_pil)
    w, h = img_pil.size

    if not result or result == [None]:
        return
    cur_result = result[0]
    # tensor to numpy
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1].cpu().numpy()
    score = cur_result[2].cpu().numpy()
    # Filter based on threshold
    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]
    # Number of instances
    num_mask = seg_label.shape[0]
    # Draw examples with larger areas first and then smaller areas later
    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    # Obtain masks based on the test results
    if num_mask > 0:
        # Save each instance detection graph and count the number of clicks per instance
        masks, shapes, cls_confs = get_masks(seg_label, num_mask, cate_label, cate_score, w, h, class_name_to_id, img)
        # Save each instance information to txt
        al_txt(cls_confs)
        vis_save(img, masks, class_name_to_id, base)
        if shapes:
            img_data = img2str(os.path.join(unlab_img_path + i))
            save_json(shapes, w, h, img_data)
        else:
            print("img:{} have no object, so no json is generated!".format(os.path.basename(i)))


def get_masks(seg_label, num_mask, cate_label, cate_score, w, h, class_name_to_id, img):
    ins_num = 1
    new_shapes = []
    det_label = {}
    masks = {}
    person_ID = 0
    rider_ID = 0
    car_ID = 0
    truck_ID = 0
    bus_ID = 0
    train_ID = 0
    motorcycle_ID = 0
    bicycle_ID = 0
    ID_dict = {}
    cls_confs = []
    for idx in range(num_mask):
        idx = -(idx + 1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        cur_mask_bool = cur_mask.astype(np.bool)
        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label = "".join([k for k, v in class_name_to_id.items() if v == cur_cate + 1])

        instance = (label, int(cur_score * 100))
        masks[instance] = cur_mask_bool

        if every_ins_viz:
            masks_ins = {instance: cur_mask_bool}
            vis_ins_save(img, masks_ins, class_name_to_id, base, ins_num, label)
            ins_num += 1
        if generate_json:
            no_occlu_flag = obj_occluded(cur_mask, (w, h))
            if no_occlu_flag:
                new_shapes, cls_confs = generate_points_no_occlu(cur_mask, (w, h), label, new_shapes, cur_score, cls_confs)
            else:
                new_shapes, det_label, person_ID, rider_ID, car_ID, truck_ID, bus_ID, train_ID, motorcycle_ID, bicycle_ID, ID_dict, cls_confs = generate_points(
                    cur_mask, (w, h), label, new_shapes, idx, det_label, person_ID, rider_ID, car_ID, truck_ID, bus_ID, train_ID, motorcycle_ID, bicycle_ID, ID_dict, cur_score, cls_confs)

    return masks, new_shapes, cls_confs


def vis_ins_save(img, masks_ins, class_name_to_id, base, ins_num, label):
    viz = img
    if masks_ins:
        labels, captions, masks = zip(
            *[
                (class_name_to_id[cnm], cnm + str(":") + str(score), msk)
                for (cnm, score), msk in masks_ins.items()
                if cnm in class_name_to_id
            ]
        )
        viz = imgviz.instances2rgb(
            image=img,
            labels=labels,
            masks=masks,
            captions=captions,
            font_size=15,
            line_width=2,
        )
    save_ins_dir = osp.join(solo_det_path, "single", base + "/")
    if not os.path.exists(save_ins_dir):
        os.makedirs(save_ins_dir)
    out_viz_file = osp.join(save_ins_dir, base + '_' + str(label) + '_' + str(ins_num) + '.jpg')
    imgviz.io.imsave(out_viz_file, viz)
    masks_ins.clear()


def obj_occluded(binary_mask, image_size):
    binary_mask = resize_binary_mask(binary_mask, image_size)
    segmentation = binary_mask_to_polygon(binary_mask, tolerance=2)
    if len(segmentation) == 1:
        no_occlu_flag = True
    else:
        no_occlu_flag = False
    return no_occlu_flag


def generate_points_no_occlu(binary_mask, image_size, label, new_shapes, cur_score, cls_confs):
    binary_mask = resize_binary_mask(binary_mask, image_size)
    segmentation = binary_mask_to_polygon(binary_mask, tolerance=2)
    new_points = []
    for i in range(0, len(segmentation[0]), 2):
        new_points.append(segmentation[0][i:i + 2])

    pt_num = len(new_points)
    cls_confs.append([label, cur_score, pt_num])

    shape_data = {
        "label": label,
        "points": new_points,
        "group_id": None,
        "shape_type": 'polygon',
        "flags": {}
    }
    new_shapes.append(shape_data)

    return new_shapes, cls_confs


def generate_points(binary_mask, image_size, label, new_shapes, idx, det_label, person_ID, rider_ID, car_ID, truck_ID, bus_ID, train_ID, motorcycle_ID, bicycle_ID, ID_dict, cur_score, cls_confs):
    binary_mask = resize_binary_mask(binary_mask, image_size)
    segmentation = binary_mask_to_polygon(binary_mask, tolerance=2)
    if label not in det_label.values():
        if label == 'person':
            person_ID = 0
            ID_dict['person'] = person_ID
        elif label == 'rider':
            rider_ID = 0
            ID_dict['rider'] = rider_ID
        elif label == 'car':
            car_ID = 0
            ID_dict['car'] = car_ID
        elif label == 'truck':
            truck_ID = 0
            ID_dict['truck'] = truck_ID
        elif label == 'bus':
            bus_ID = 0
            ID_dict['bus'] = bus_ID
        elif label == 'train':
            train_ID = 0
            ID_dict['train'] = train_ID
        elif label == 'motorcycle':
            motorcycle_ID = 0
            ID_dict['motorcycle'] = motorcycle_ID
        elif label == 'bicycle':
            bicycle_ID = 0
            ID_dict['bicycle'] = bicycle_ID

        det_label[idx] = label

    elif label in det_label.values():
        if label == 'person':
            person_ID += 1
            ID_dict['person'] = person_ID
        elif label == 'rider':
            rider_ID += 1
            ID_dict['rider'] = rider_ID
        elif label == 'car':
            car_ID += 1
            ID_dict['car'] = car_ID
        elif label == 'truck':
            truck_ID += 1
            ID_dict['truck'] = truck_ID
        elif label == 'bus':
            bus_ID += 1
            ID_dict['bus'] = bus_ID
        elif label == 'train':
            train_ID += 1
            ID_dict['train'] = train_ID
        elif label == 'motorcycle':
            motorcycle_ID += 1
            ID_dict['motorcycle'] = motorcycle_ID
        elif label == 'bicycle':
            bicycle_ID += 1
            ID_dict['bicycle'] = bicycle_ID

        det_label[idx] = label
    pt_nums = 0
    for points in segmentation:
        new_points = []
        for i in range(0, len(points), 2):
            new_points.append(points[i:i + 2])
        pt_num = len(new_points)
        pt_nums += pt_num

        shape_data = {
            "label": label,
            "points": new_points,
            "group_id": ID_dict[label],
            "shape_type": 'polygon',
            "flags": {}
        }
        new_shapes.append(shape_data)

    cls_confs.append([label, cur_score, pt_nums])

    return new_shapes, det_label, person_ID, rider_ID, car_ID, truck_ID, bus_ID, train_ID, motorcycle_ID, bicycle_ID, ID_dict, cls_confs


def al_txt(cls_confs):
    if cls_confs is not None:
        with open(os.path.join(solo_det_path, "al_info", base + '.txt'), 'w') as f:
            for i in cls_confs:
                line = str(i).replace("'", '')
                line = line.replace("[]", '')
                line = line.strip("[]")
                f.write(line + '\n')
        f.close()


def vis_save(img, masks, class_name_to_id, base):
    viz = img
    if masks:
        labels, captions, masks = zip(
            *[
                (class_name_to_id[cnm], cnm + str(":") + str(score), msk)
                for (cnm, score), msk in masks.items()
                if cnm in class_name_to_id
            ]
        )
        viz = imgviz.instances2rgb(
            image=img,
            labels=labels,
            masks=masks,
            captions=captions,
            font_size=15,
            line_width=2,
        )
    out_viz_file = osp.join(solo_det_path, "whole", base + "_out.jpg")
    imgviz.io.imsave(out_viz_file, viz)
    if every_ins_viz:
        copyfile(out_viz_file, osp.join(solo_det_path, "single", base, base + "_out.jpg"))


def img2str(image_name):
    with open(image_name, "rb") as file:
        base64_data = str(base64.b64encode(file.read()))
    match_pattern = re.compile(r"b'(.*)'")
    base64_data = match_pattern.match(base64_data).group(1)
    return base64_data


def save_json(shapes, w, h, img_data):
    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": i,
        "imageData": img_data,
        "imageHeight": h,
        "imageWidth": w
    }
    f = open(os.path.join(solo_det_path, "json", base + '.json'), 'w')
    json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    network = Network()

    for i in os.listdir(unlab_img_path):
        base = osp.splitext(osp.basename(i))[0]
        start_time = time.time()
        segment_results = network.segmentation(unlab_img_path + i)
        end_time = time.time()
        t = end_time - start_time
        print("img :{} det time: {}秒, FPS={}".format(os.path.basename(i), round(t, 2), round(1 / t, 1)))
        main(i, segment_results, class_name_to_id, score_thr=det_thr, sort_by_density=True)
