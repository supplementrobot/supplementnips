# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images

import torch
import torch.nn as nn
import torchvision.models as TVmodels
# from TV_offline_models.swin_transformer import swin_v2_t,swin_v2_s
import MinkowskiEngine as ME

from models.minkloc import MinkLoc
from network.resnetfpn_simple import ImageGeM

from PIL import Image

import os
import tqdm


import numpy as np


import pickle

from tools.utils import set_seed
set_seed(7)
from tools.options import Options
args = Options().parse()




def create_lidar2img_ndx(lidar_timestamps, image_timestamps, k, threshold):
    print('Creating lidar2img index...')
    delta_l = []
    lidar2img_ndx = {}
    count_above_thresh = 0

    for lidar_ts in tqdm.tqdm(lidar_timestamps):
        nn_ndx = find_k_closest(image_timestamps, lidar_ts, k)
        nn_ts = image_timestamps[nn_ndx]
        nn_dist = np.abs(nn_ts - lidar_ts)
        #assert (nn_dist <= threshold).all(), 'nn_dist: {}'.format(nn_dist / 1000)
        # threshold is in miliseconds
        if (nn_dist > threshold*1000).sum() > 0:
            count_above_thresh += 1

        # Remember timestamps of closest images
        lidar2img_ndx[lidar_ts] = list(nn_ts)
        delta_l.extend(list(nn_dist))

    delta_l = np.array(delta_l, dtype=np.float32)
    s = 'Distance between the lidar scan and {} closest images (min/mean/max): {} / {} / {} [ms]'
    print(s.format(k, int(np.min(delta_l)/1000), int(np.mean(delta_l)/1000), int(np.max(delta_l)/1000)))
    print('{} scans without {} images within {} [ms] threshold'.format(count_above_thresh, k, int(threshold)))

    return lidar2img_ndx



def find_k_closest(arr, x, k):
    # This function returns indexes of k closest elements to x in arr[]
    n = len(arr)
    # Find the crossover point
    l = find_cross_over(arr, 0, n - 1, x)
    r = l + 1  # Right index to search
    ndx_l = []

    # If x is present in arr[], then reduce left index. Assumption: all elements in arr[] are distinct
    if arr[l] == x:
        l -= 1

    # Compare elements on left and right of crossover point to find the k closest elements
    while l >= 0 and r < n and len(ndx_l) < k:
        if x - arr[l] < arr[r] - x:
            ndx_l.append(l)
            l -= 1
        else:
            ndx_l.append(r)
            r += 1

    # If there are no more elements on right side, then print left elements
    while len(ndx_l) < k and l >= 0:
        ndx_l.append(l)
        l -= 1

    # If there are no more elements on left side, then print right elements
    while len(ndx_l) < k and r < n:
        ndx_l.append(r)
        r += 1

    return ndx_l





def find_cross_over(arr, low, high, x):
    if arr[high] <= x:  # x is greater than all
        return high

    if arr[low] > x:  # x is smaller than all
        return low

    # Find the middle point
    mid = (low + high) // 2  # low + (high - low)// 2

    # If x is same as middle element, then return mid
    if arr[mid] <= x and arr[mid + 1] > x:
        return mid

    # If x is greater than arr[mid], then either arr[mid + 1] is ceiling of x or ceiling lies in arr[mid+1...high]
    if arr[mid] < x:
        return find_cross_over(arr, mid + 1, high, x)

    return find_cross_over(arr, low, mid - 1, x)






if __name__ == '__main__':





    scenes_list = [
        '2014-07-14-14-49-50',
        '2014-11-18-13-20-12',
        '2014-12-02-15-30-08',
        '2014-12-09-13-21-02',
        '2014-12-12-10-45-15',

        '2015-02-03-08-45-10',
        '2015-02-13-09-16-26',
        '2015-03-10-14-18-10',
        '2015-05-19-14-06-38',
        '2015-08-13-16-02-58',

    ]



    lidars_root = '/home/user/vpr/benchmark_datasets/oxfordadafusion'
    
    images_root = '/home/user/vpr/RobotCar_checked_image_adafusion'


    # RUNS_FOLDER = "oxfordadafusion/"
    # FILENAME = "pointcloud_locations_20m_10overlap.csv"
    # POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"



    lidar2image_ndx_pickle = {}

    total_length = 0



    lidar_timestamps = []

    image_timestamps = []


    for scene_name in scenes_list:
        lidar_scene_folder = os.path.join(lidars_root, scene_name)
        lidars_folder = os.path.join(lidar_scene_folder,'pointcloud_20m_10overlap')
        lidars_list = os.listdir(lidars_folder)
        lidars_list = sorted(lidars_list)
        lidar_timestamps.extend(lidars_list)


        image_scene_folder = os.path.join(images_root, scene_name)
        images_list = os.listdir(image_scene_folder)
        images_list = sorted(images_list)
        image_timestamps.extend(images_list)

        

    lidar_timestamps = [int(lidar_timestamp.replace('.bin','')) for lidar_timestamp in lidar_timestamps]
    lidar_timestamps = sorted(lidar_timestamps)
    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)

    image_timestamps = [int(image_timestamp.replace('.png','')) for image_timestamp in image_timestamps]
    image_timestamps = sorted(image_timestamps)
    image_timestamps = np.array(image_timestamps, dtype=np.int64)



    k = 20
    nn_threshold = 1000


    # Find k closest images for each lidar timestamp
    lidar2img_ndx_pickle = 'oxfordadafusion_lidar2image_ndx.pickle'
    ndx_filepath = os.path.join(images_root, 'oxfordadafusion_lidar2image_ndx.pickle')

    if os.path.exists(ndx_filepath):
        with open(ndx_filepath, 'rb') as f:
            lidar2img_ndx = pickle.load(f)
    else:
        lidar2img_ndx = create_lidar2img_ndx(lidar_timestamps, image_timestamps, k, nn_threshold)
        with open(ndx_filepath, 'wb') as f:
            pickle.dump(lidar2img_ndx, f)


    a=1