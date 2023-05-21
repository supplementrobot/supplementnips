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
from tqdm import tqdm

import pickle

from tools.utils import set_seed
set_seed(7)
from tools.options import Options
args = Options().parse()







if __name__ == '__main__':




    # benchmark_root = '/home/user/vpr/BenchmarkBoreas/'
    # benchmark_root = '/home/user/vpr/BenchmarkBoreasv2/'
    benchmark_root = '/home/user/vpr/BenchmarkBoreasv3/'
    



    # scenes_list = [
    #     'boreas-2020-11-26-13-58', # snow
    #     'boreas-2020-12-01-13-26', # snowing
    #     'boreas-2020-12-18-13-44', # sun snow
    #     'boreas-2021-05-13-16-11', # sun


    #     'boreas-2021-01-26-11-22', # snowing
    #     'boreas-2021-04-29-15-55', # raining

    #     'boreas-2021-06-17-17-52', # sun
        
    #     'boreas-2021-09-14-20-00'  # night
    # ]


    scenes_list = [
        'boreas-2020-11-26-13-58', # snow
        'boreas-2020-12-01-13-26', # snowing
        'boreas-2020-12-18-13-44', # sun snow
        'boreas-2021-05-13-16-11', # sun

        'boreas-2021-07-20-17-33', # clouds rain
        'boreas-2021-07-27-14-43', # clouds



        'boreas-2021-01-26-11-22', # snowing
        'boreas-2021-04-29-15-55', # raining
        'boreas-2021-06-17-17-52', # sun
        'boreas-2021-09-14-20-00',  # night

        'boreas-2021-10-15-12-35', # clouds
        'boreas-2021-11-02-11-16', # sun clouds
    ]




    lidar2image_ndx_pickle = {}

    total_length = 0
    for scene_name in scenes_list:

        scene_folder = os.path.join(benchmark_root, 'boreas', scene_name)

        lidars_folder = os.path.join(scene_folder,'lidar_1_4096_interval10')
        lidars_list = os.listdir(lidars_folder)
        lidars_list = sorted(lidars_list)

        images_folder = os.path.join(scene_folder,'camera_lidar_interval10')
        images_list = os.listdir(images_folder)
        images_list = sorted(images_list)

        assert len(images_list) == len(lidars_list)
        total_length += len(images_list)


        for i, lidar_name in tqdm(enumerate(lidars_list)):
            lidar_name_pure = int(lidar_name.replace('.npy',''))
            image_name_pure = int(images_list[i].replace('.png',''))
            # image = Image.open(os.path.join(images_folder,str(image_name_pure)+'.png')).convert('RGB')
            # image.save(os.path.join(images_folder,str(image_name_pure)+'.png'))

            # lidar2image_ndx_pickle[lidar_name_pure:[image_name_pure]]
            lidar2image_ndx_pickle.update({lidar_name_pure:[image_name_pure]})

    assert len(lidar2image_ndx_pickle) == total_length


    ndx_filepath = os.path.join(benchmark_root,'boreas_lidar2image_ndx.pickle')
    with open(ndx_filepath, 'wb') as f:
        pickle.dump(lidar2image_ndx_pickle, f)


