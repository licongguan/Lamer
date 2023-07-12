# -*- coding: utf-8 -*
import os
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2
from scipy import linalg
import os.path as osp
from mmautoeval.cal_map import main_map


def main_fd_solo(path, spec_label):
    config = '/media/glc/Elements/project/SOLO/MyTrain/ALSOLO/Public-cityspace/01_AL/solov2_Public_cityspace_01_AL.py'
    checkpoint = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace/01_AL/work_dirs/latest.pth'
    cuda_device = 'cuda:0'
    dims = 2048
    img_resize_shape = (64, 32)

    assert img_resize_shape[0] * img_resize_shape[1] == dims, 'shape error'

    if cuda_device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(cuda_device)

    # cal fd
    fid_value = calculate_AP_fid_given_paths(path, device, config, checkpoint, dims, img_resize_shape, spec_label)

    return fid_value


def calculate_AP_fid_given_paths(paths, device, config, checkpoint, dims, img_resize_shape, spec_label):
    model = init_detector(config, checkpoint, device)

    m1, s1 = compute_stat_of_path(paths[0], model, dims, img_resize_shape, spec_label)
    m2, s2 = compute_stat_of_path(paths[1], model, dims, img_resize_shape, spec_label)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def compute_stat_of_path(path, model, dims, img_resize_shape, spec_label):
    files = os.listdir(path)
    files = [os.path.join(path, filename) for filename in files]

    act = get_activat(path, files, model, dims, img_resize_shape, spec_label)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


def get_activat(path, files, model, dims, img_resize_shape, spec_label):
    spec_nums = 0
    for item in files:
        img = cv2.imread(item)
        img_resize = cv2.resize(img, img_resize_shape)
        result = inference_detector(model, img_resize)
        # 检测到指定类别的num
        spec_num = get_result_num(result, spec_label, score_thr=0.1)
        spec_nums += spec_num
    # all img have car num
    labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    print("{} have detect {} = {}".format(osp.basename(osp.normpath(path)), labels[spec_label], spec_nums))

    pred_arr = np.empty((spec_nums, dims))

    start_idx = 0
    for item in files:
        img = cv2.imread(item)
        img_resize = cv2.resize(img, img_resize_shape)
        result = inference_detector(model, img_resize)
        pred, spec_num = filter_result(result, spec_label, dims, score_thr=0.1)
        # append
        pred_arr[start_idx:start_idx + spec_num] = pred
        start_idx = start_idx + spec_num

    return pred_arr


def get_result_num(result, spec_label, score_thr=0.1):
    spec_num = 0
    if result:
        cur_result = result[0]
        if cur_result is None:
            spec_num = 0
        else:
            cate_label = cur_result[1]
            cate_label = cate_label.cpu().numpy()
            score = cur_result[2].cpu().numpy()
            # all index
            vis_inds = score > score_thr
            cate_label = cate_label[vis_inds]

            # spec car index
            spec_label_index = np.where(cate_label == spec_label)
            spec_num = len(cate_label[spec_label_index])

    return spec_num


def filter_result(result, spec_label, dims, score_thr=0.1):
    if result:
        cur_result = result[0]
        if cur_result is None:
            spec_num = 0
            pred = np.zeros((1, dims))
            return pred, spec_num

        else:
            seg_label = cur_result[0]
            seg_label = seg_label.cpu().numpy().astype(np.uint8)
            cate_label = cur_result[1]
            cate_label = cate_label.cpu().numpy()
            score = cur_result[2].cpu().numpy()
            # all index
            vis_inds = score > score_thr
            seg_label = seg_label[vis_inds]
            cate_label = cate_label[vis_inds]

            # spec car index
            spec_label_index = np.where(cate_label == spec_label)
            spec_num = len(cate_label[spec_label_index])
            if spec_num == 0:
                spec_seg_label = np.zeros((1, dims))
            else:
                spec_seg_label = seg_label[spec_label_index].reshape((spec_num, dims))

            pred = spec_seg_label
            # print(pred)

            return pred, spec_num


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':
    for i in range(36):
        group = '0{}'.format(i + 1)
        path_1 = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace/01_AL/FD/aachen/'
        path_2 = '/media/glc/Elements/DATA/ALSOLO/Public-cityspace/{}_AL/FD/'.format(group)
        for item in os.listdir(path_2):
            path_2 = path_2 + item + "/"
        path = [path_1, path_2]

        spec_label = 0
        labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        my_fd_solo = main_fd_solo(path, spec_label)
        print("{} and {} FD: ".format(osp.basename(osp.normpath(path[0])), osp.basename(osp.normpath(path[1]))),
              round(my_fd_solo, 3))
