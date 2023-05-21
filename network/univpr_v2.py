import torch
import torch.nn as nn
from network.resnetfpn_simple import *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
from network.pooling import *
import MinkowskiEngine as ME


from network.general_minkfpn import GeneralMinkFPN

from models.minkloc import MinkLoc

from network.later_cloud_branch import LaterCloudBranch
import spconv.pytorch as spconv
from layers.pooling import MinkGeM as MinkGeM
try:
    from viz_lidar_mayavi_open3d import *
except:
    None
from network.general_imagefes import GeneralImageFE

from network.ffb import FFB
from network.gatt_block import GattBlock
# from network.pointnet_simple import PointNetSimple
from pointnet2 import pointnet2_utils

from network.grids import Grids
from network.later_image_branch import LaterImageBranch

import torchvision

import matplotlib.pyplot as plt

from datasets.oxford import transform_pts_in_camsystem, project_onto_image

from network.ffb_local import FFBLocal

from torch_geometric.nn import voxel_grid


from tools.options import Options
from tools.utils import set_seed
set_seed(7)
args = Options().parse()







def fps(xyz, npoint):
    xyz_flipped = xyz.permute(0,2,1).contiguous()
    new_xyz = pointnet2_utils.gather_operation(
        xyz_flipped,
        pointnet2_utils.furthest_point_sample(xyz, npoint)
    ).transpose(1, 2).contiguous()

    return new_xyz




def fps_ball_query(xyz, npoint, radius, nsample):

    xyz_flipped = xyz.permute(0,2,1).contiguous()
    new_xyz = pointnet2_utils.gather_operation(
        xyz_flipped,
        pointnet2_utils.furthest_point_sample(xyz, npoint)
    ).transpose(1, 2).contiguous()

    model = pointnet2_utils.QueryAndGroup(radius=radius,nsample=nsample)
    output = model(xyz, new_xyz)
    
    return output






def convert_ME_to_spconv_tensor(cloud_feat, tensor_stride):
    assert isinstance(cloud_feat, ME.SparseTensor)
    # test spconv
    spconv_feats = cloud_feat.F
    spconv_coords = cloud_feat.C
    spconv_coords[:,1:] = spconv_coords[:,1:] // tensor_stride
    spatial_shape = [
        torch.max(spconv_coords[:,1]) - torch.min(spconv_coords[:,1]), 
        torch.max(spconv_coords[:,2]) - torch.min(spconv_coords[:,2]), 
        torch.max(spconv_coords[:,3]) - torch.min(spconv_coords[:,3]), 
    ]
    batch_size = len(cloud_feat.decomposed_coordinates)
    cloud_feat_spconv = spconv.SparseConvTensor(spconv_feats, spconv_coords, spatial_shape, batch_size=batch_size)

    return cloud_feat_spconv




def obtain_same_length_cloud_features(cloud_feat):
    assert isinstance(cloud_feat, ME.SparseTensor) or isinstance(cloud_feat, spconv.SparseConvTensor)
    
    if isinstance(cloud_feat, ME.SparseTensor):
        # # -- obtain same length cloud features
        # -- fps ball query
        # clouds, cloud_feat = self.coordsclouds(cloud_feat, data_dict)
        # clouds, cloud_feat = self.pointnet2mlp(clouds, cloud_feat) # [b,npoint,3]  [b,c,npoint]
        # cloud_feat_gem = self.cloud_imagegem(cloud_feat.unsqueeze(-1))
        # -- 
        cloud_feat_list = cloud_feat.decomposed_features

    elif isinstance(cloud_feat, spconv.SparseConvTensor):
        # test spconv same length
        cloud_feat_list = []
        for i in range(cloud_feat.batch_size):
            rows_of_this_batch = cloud_feat.features[cloud_feat.indices[:,0]==i]
            cloud_feat_list.append(rows_of_this_batch)



    # lengths_list = [len(row) for row in cloud_feat_list]
    # # # -- 1
    # cloud_feat = torch.nn.utils.rnn.pad_sequence(cloud_feat_list, batch_first=True)
    # # -- 2
    cloud_feat_same_length = []
    for i in range(len(cloud_feat_list)):
        this_x = cloud_feat_list[i]
        if len(this_x) < args.num_cloud_points:
            is_replace = True
        else:
            is_replace = False

        selected_indices = np.random.choice(len(this_x), size=args.num_cloud_points, replace=is_replace)
        new_x = this_x[selected_indices]
        cloud_feat_same_length.append(new_x)

    cloud_feat_same_length = torch.stack(cloud_feat_same_length) # [b,npts,c]

    return cloud_feat_same_length












class UniVPRV2(nn.Module):
    def __init__(self):
        super(UniVPRV2, self).__init__()







        self.image_fe = GeneralImageFE(image_fe=args.image_fe, 
                                       num_other_stage_blocks=args.num_other_stage_blocks,
                                       num_stage3_blocks=args.num_stage3_blocks,
                                       input_type='image')
        

        if args.sph_cloud_fe is not None:
            self.sph_cloud_fe = GeneralImageFE(image_fe=args.sph_cloud_fe, 
                                        num_other_stage_blocks=args.sph_num_other_stage_blocks,
                                        num_stage3_blocks=args.sph_num_stage3_blocks,
                                        input_type='sph_cloud')




        if args.cloud_fe == 'general_minkfpn':
            self.cloud_fe = GeneralMinkFPN(in_channels=1, out_channels=None, num_top_down=None, conv0_kernel_size=5,
                                        layers=[1,1,1], planes=[32,64,64])
        elif args.cloud_fe == 'minkloc':
            self.cloud_fe = MinkLoc(in_channels=1, feature_size=128, output_dim=128,
                            planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                            conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')

    


        self.ffbs = nn.ModuleList()
        self.laterimagebranches = nn.ModuleList()
        self.latercloudbranches = nn.ModuleList()
        self.ffblocals = nn.ModuleList()

        for i in range(args.num_ffbs):
            ffb = FFB(in_features=args.fusion_dim, hidden_features=args.fusion_dim//8, n_heads=8, 
                      num_blocks=args.num_blocks_in_ffb, 
                        ffb_type=args.ffb_type,
                        num_image_points=args.num_image_points, num_cloud_points=args.num_cloud_points,
                        image_fe_last_dim=self.image_fe.last_dim)
            laterimagebranch = LaterImageBranch(block_type=args.later_image_branch_type,num_heads=8,
                                            in_features=self.image_fe.last_dim, 
                                            num_blocks=args.num_blocks_in_later_image,
                                            window_size=[7,7], shift_size=[0,0])
            latercloudbranch = LaterCloudBranch(in_features=128,
                                                num_blocks=args.num_blocks_in_later_cloud)
            # ffblocal = FFBLocal(in_image_feat_dim=self.image_fe.last_dim, embed_dim=args.fusion_dim, num_heads=8, num_blocks=1, window_size=4)
            ffblocal = FFBLocal(in_image_feat_dim=self.image_fe.last_dim, embed_dim=args.fusion_dim, num_heads=8, num_blocks=1, window_size=args.ffblocal_windowsize)


            self.ffbs.append(ffb)
            self.laterimagebranches.append(laterimagebranch)
            self.latercloudbranches.append(latercloudbranch)
            self.ffblocals.append(ffblocal)


     
    def forward(self, data_dict):
        # feed_dict{
        #     'positives_mask(train)',
        #     'negatives_mask(train)',
        #     'coords',
        #     'clouds',
        #     'features',
        #     'uv',
        #     'colors',
        #     'maskselect',
        #     'images',
        #     'P0_camera',
        #     'T_camera_lidar_basedon_pose',
        # }


    



        # -- image part
        image_feat_128, image_feat_gem_org, image_feat_256 = self.image_fe(data_dict)








        # -- cloud part
        cloud_feat_dict = self.cloud_fe(data_dict)
        cloud_feat = cloud_feat_dict['x_feat']
        cloud_feat_gem_org = cloud_feat_dict['embedding']
        cloud_feat_gem_extra = cloud_feat_dict['embedding_extra']



        # _coords = cloud_feat.coordinates[cloud_feat.coordinates[:,0]==0][:,1:]
        # T_camera_lidar_basedon_pose = data_dict['T_camera_lidar_basedon_pose'][0]
        # P0_camera = data_dict['P0_camera'][0]
        # _pts_in_image, _pts_uv_in_image, _pts_colors_in_image, _pts_out_image = convert_pts_to_uv(_coords, T_camera_lidar_basedon_pose, P0_camera)
        # _image = data_dict['images'][0]
        # _pts_uv_in_image = _pts_uv_in_image * 0.125 / 1 # 16 is the downsample ratio of the image feature
        # _pts_window_indices, _image, _image_window_indices = get_window_indices(_pts_uv_in_image, _image, window_size=[64,64])

        # _pts_in_image = _pts_in_image.cpu().numpy()
        # _pts_uv_in_image = _pts_uv_in_image.cpu().numpy()
        # _pts_colors_in_image = _pts_colors_in_image.cpu().numpy()

        # # _pts_all = torch.cat([_pts_in_image, _pts_out_image], dim=0).cpu().numpy()
        # # viz_lidar_open3d(_pts_all)
        # # _coords = cloud_feat.coordinates[cloud_feat.coordinates[:,0]==0][:,1:]
        # # _coords = _coords.cpu().numpy()
        # # # _clouds = _clouds.cpu().numpy()
        # # # viz_lidar_open3d(_clouds)
        # # _T = data_dict['T_camera_lidar_basedon_pose'][0].cpu().numpy()
        # # _P0 = data_dict['P0_camera'][0].cpu().numpy()
        # # _coordsincamsystem = transform_pts_in_camsystem(_coords,T=_T)
        # # # viz_lidar_open3d(_coordsincamsystem)
        # # _uv,_colors,_mask = project_onto_image(_coordsincamsystem,P=_P0)
        # # _selected_indices = (_coords[:,0]>0) & (_coords[:,0]<1000) \
        # #             & (_coords[:,1]>0) & (_coords[:,1]<1000) \
        # #             & (_coords[:,2]>-10) & (_coords[:,2]<50)
        # # _maskselect = _mask & _selected_indices
        # # # _uv = _uv[_maskselect]
        # # _uv = _uv * 0.125 / 16
        # # viz projected points  
        # plt.figure(dpi=200)
        # # _image = image_feat_128[0,:3].cpu()
        # # _image = image_feat_256[0,:3].cpu()
        # _image = data_dict['images'][0].cpu()
        # _mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
        # _std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
        # _image = _image*_std + _mean
        # _image = torchvision.transforms.ToPILImage()(_image)
        # _image = np.array(_image)
        # # _uv = _uv[_maskselect]
        # # _colors = _colors[_maskselect]
        # _uv = _pts_uv_in_image  * 0.125 
        # h_max = _uv[:,1].max()
        # w_max = _uv[:,0].max()

        # _colors = _pts_colors_in_image
        # plt.scatter(_uv[:,0], _uv[:,1], c=_colors, s=1)
        # plt.imshow(_image)
        # plt.show()
        # # plt.close()


        a=1



        # -- lidar_sph
        if hasattr(self, 'sph_cloud_fe'):
            sph_cloud_feat, sph_cloud_feat_gem_org, sph_cloud_feat_256 = self.sph_cloud_fe(data_dict)






        if args.num_ffbs > 0:

            for i in range(args.num_ffbs):
                ffb = self.ffbs[i]
                laterimagebranch = self.laterimagebranches[i]
                latercloudbranch = self.latercloudbranches[i]
                ffblocal = self.ffblocals[i]



                cloud_feat_same_length = obtain_same_length_cloud_features(cloud_feat)
                

                output_dict = ffb(in_image_feat=image_feat_128, 
                                    in_cloud_feat=cloud_feat_same_length)
                ffb_output_gem_128 = output_dict['output_feat_gem_128']
                ffb_output_gem_256 = output_dict['output_feat_gem_256']



                image_feat_128, image_feat_256_gem_later, image_feat_256 = laterimagebranch(image_feat_256, ffb_output_gem_256)
                cloud_feat, cloud_feat_gem_later = latercloudbranch(cloud_feat, ffb_output_gem_128)

                if args.use_ffblocal:
                    image_feat_256, cloud_feat = ffblocal(image_feat_256, cloud_feat, data_dict['T_camera_lidar_basedon_pose'], data_dict['P0_camera'], 
                                                        downsample_ratio=16, image=data_dict['images'])



            if args.final_embedding_type == 'imageorg_cloudorg_ffb':
                embedding = torch.cat([image_feat_gem_org, cloud_feat_gem_org, ffb_output_gem_128], dim=1)
            elif args.final_embedding_type == 'imagelater_cloudlater_ffb':
                embedding = torch.cat([image_feat_256_gem_later, cloud_feat_gem_later, ffb_output_gem_128], dim=1)
            elif args.final_embedding_type == 'imagelater_cloudlater':
                embedding = torch.cat([image_feat_256_gem_later, cloud_feat_gem_later], dim=1)
            elif args.final_embedding_type == 'imageorg':
                embedding = image_feat_gem_org
            elif args.final_embedding_type == 'cloudorg':
                embedding = cloud_feat_gem_org
            elif args.final_embedding_type == 'imagelater_cloudlater_ffb_sphcloud':
                embedding = torch.cat([image_feat_256_gem_later, cloud_feat_gem_later, ffb_output_gem_128, sph_cloud_feat_gem_org], dim=1)
            else:
                raise Exception





        elif args.num_ffbs == 0:


            embedding = torch.cat([image_feat_gem_org, cloud_feat_gem_org], dim=1)




        

        # print(f'beforfusion_{args.use_l2norm_before_fusion}_outputfeat_{args.use_l2norm_for_output_feat}')


        output_dict = {
            'image_embedding':image_feat_gem_org,
            'cloud_embedding':cloud_feat_gem_org,
            'embedding':embedding,
        }

        return output_dict