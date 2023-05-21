import argparse
import os



class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):



        self.parser.add_argument('--cuda', type=str, default='0')
        self.parser.add_argument('--tryid', type=int, default=2)
        self.parser.add_argument('--save_weights', type=str, default=True)
        self.parser.add_argument('--num_workers', type=int, default=8)
        # oxford   boreas
        self.parser.add_argument('--dataset', type=str, 
                                #  default='oxford',
                                #  default='oxfordadafusion',
                                 default='boreas'
                                 )
        self.parser.add_argument('--dataset_folder', type=str, 
                                 default='/data/user/nips2023_supplement/BenchmarkBoreasv3',
                                 )
        self.parser.add_argument('--image_path', type=str, 
                                 default='/data/user/nips2023_supplement/BenchmarkBoreasv3/boreas',
                                 )
        self.parser.add_argument('--n_points_boreas', type=int, default=4096) # only for boreas



        # False True
        self.parser.add_argument('--use_minkloc', type=str, default=False)
        # resnet18org  resnet34org
        # resnet18   resnet34   convnext_tiny  convnext_small  swin_t  swin_s  swin_v2_t  swin_v2_s
        self.parser.add_argument('--minkloc_image_fe', type=str, default='resnet18org')
        # minkfpn  generalminkfpn
        self.parser.add_argument('--minkfpn', type=str, default='minkfpn')



        self.parser.add_argument('--use_ffblocal', type=str, default=True)

        
        self.parser.add_argument('--ffblocal_windowsize', type=int, default=1)
        self.parser.add_argument('--use_proj_inffblocal', type=str, default=False)
        self.parser.add_argument('--usemeconv1x1_beforesptr', type=str, default=False)
        self.parser.add_argument('--usemeconv1x1_aftersptr', type=str, default=True)
        # conv  fc  fcode  fcode64
        self.parser.add_argument('--beforeafter_convtype', type=str, default='fcode')
        self.parser.add_argument('--beforeafter_useres', type=str, default=False)     # only for fcode
        self.parser.add_argument('--beforeafter_useextrafc', type=str, default=False) # only for fcode
        self.parser.add_argument('--beforeafter_sharefc', type=str, default=False) # only for fcode

        self.parser.add_argument('--image_useswin', type=str, default=True) 
        self.parser.add_argument('--imageswin_windowsize', type=int, default=4) 
        self.parser.add_argument('--imageswin_useproj', type=str, default=False) 
        self.parser.add_argument('--imageswin_userelu', type=str, default=False) 
        # swin   swinode 
        self.parser.add_argument('--imageswin_type', type=str, default='swinode') 
        self.parser.add_argument('--imageswin_useres', type=str, default=False) 






        # resnet18   resnet34  resnet50  
        # convnext_tiny      convnext_small  
        # swin_t    swin_s    swin_v2_t    swin_v2_s  [batch60]
        # efficientnet_b0   efficientnet_b1   efficientnet_b2   efficientnet_v2_s
        # regnet_x_3_2gf    regnet_y_1_6gf    regnet_y_3_2gf
        self.parser.add_argument('--image_fe', type=str, default='resnet18')
        # convnext_tiny[3,3,9,3]  
        self.parser.add_argument('--num_other_stage_blocks', type=int, default=3)
        self.parser.add_argument('--num_stage3_blocks', type=int, default=3)
        self.parser.add_argument('--sph_cloud_fe', type=str, default=None) # None 'resnet18'
        # self.parser.add_argument('--sph_num_other_stage_blocks', type=int, default=3)
        # self.parser.add_argument('--sph_num_stage3_blocks', type=int, default=3)

        # general_minkfpn   minkloc  
        self.parser.add_argument('--cloud_fe', type=str, default='minkloc')
        self.parser.add_argument('--num_image_points', type=int, default=300)
        self.parser.add_argument('--num_cloud_points', type=int, default=128)
        self.parser.add_argument('--fusion_dim', type=int, default=128)
        # ln
        self.parser.add_argument('--gattnorm', type=str, default='ln')
        # gelu  relu
        self.parser.add_argument('--gattactivation', type=str, default='relu')
        # 4096  6144  8192   




        self.parser.add_argument('--num_ffbs', type=int, default=2)
        self.parser.add_argument('--num_blocks_in_later_image', type=int, default=1)
        self.parser.add_argument('--num_blocks_in_later_cloud', type=int, default=1)
        self.parser.add_argument('--num_blocks_in_ffb', type=int, default=1) # exclude the first block
        self.parser.add_argument('--use_l2norm_before_fusion', type=str, default=True)
        self.parser.add_argument('--use_a_fusion_first', type=bool, default=True)
        # gemextra   randomsamelength   gemextra_randomsamelength
        self.parser.add_argument('--in_cloud_feat_type', type=str, default='randomsamelength')
        # pure   mix   imageonly
        self.parser.add_argument('--ffb_type', type=str, default='pure')
        # basicblock  bottleneck  gatt  gattm   attn  resattn   swinblock
        self.parser.add_argument('--gatt_image_block_type', type=str, default='swinblock')
        # gatt  gattm   attn  resattn
        self.parser.add_argument('--gatt_cloud_block_type', type=str, default='attn')


        # gatt  gattm   attn  resattn
        # qkv   qkvm   qkvg1   qkvg2
        self.parser.add_argument('--use_attn2', type=str, default=True)
        self.parser.add_argument('--use_attn3', type=str, default=True)
        self.parser.add_argument('--use_attninlocalffb', type=str, default=True)
        self.parser.add_argument('--gatt_fusion_block_type', type=str, default='qkvg1')
        self.parser.add_argument('--gatt_fusion_block_type2', type=str, default='qkvg1')
        self.parser.add_argument('--gatt_fusion_block_type3', type=str, default='qkvg1')


        # basicblock  bottleneck  gatt  gattm   attn  resattn   swinblock
        self.parser.add_argument('--later_image_branch_type', type=str, default='basicblock')
        # basicblock  submbasicblock
        self.parser.add_argument('--later_cloud_branch_type', type=str, default='basicblock')
        # add     add_w   times_sigmoid  times_wsigmoid   None
        self.parser.add_argument('--later_image_branch_interaction_type', type=str, default='add')
        self.parser.add_argument('--later_cloud_branch_interaction_type', type=str, default='add')
        # fconfusion   fconfusiongem
        self.parser.add_argument('--make_image_fusion_same_channel', type=str, default='fconfusiongem')
        # imageorg_cloudorg_ffb   imagelater_cloudlater_ffb   imagelater_cloudlater
        # imageorg    cloudorg
        # imagelater_cloudlater_ffb_sphcloud
        self.parser.add_argument('--final_embedding_type', type=str, default='imagelater_cloudlater_ffb')
        # image   cloud   cat
        self.parser.add_argument('--useminkloc_final_embedding_type', type=str, default='cat')






        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--train_batch_size', type=int, default=80)
        self.parser.add_argument('--val_batch_size', type=int, default=160)
        self.parser.add_argument('--image_lr', type=float, default=1e-4)
        self.parser.add_argument('--cloud_lr', type=float, default=1e-3)
        self.parser.add_argument('--fusion_lr', type=float, default=1e-4)




        # -- image augmentation rate
        self.parser.add_argument('--bcs_aug_rate', type=float, default=0.2) # 0.2
        self.parser.add_argument('--hue_aug_rate', type=float, default=0.1) # 0.1



        self.parser.add_argument('--config', type=str, default='config/config_baseline_multimodal.txt')
        self.parser.add_argument('--model_config', type=str, default='models/minklocmultimodal.txt')
        self.parser.add_argument('--mink_quantization_size', type=float, default=0.01)






        self.parser.add_argument('--resume_epoch', type=int, default=-1)




        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')



        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')


        self.parser.add_argument('--debug', type=str, default=False)

        





    def parse(self):
        self.initialize()
        self.args = self.parser.parse_args()

        args_dict = vars(self.args)
        # print(args_dict)
        for k, v in args_dict.items():
            if v=='False':
                args_dict[k] = False
            elif v=='True':
                args_dict[k] = True
            elif v=='None':
                args_dict[k] = None

        self.args = argparse.Namespace(**args_dict)




        
        self.args.exp_name = ''
        self.args.exp_name += f'{self.args.tryid}'

        # -- tune swin
        if self.args.use_minkloc:
            self.args.exp_name += f'__{self.args.dataset}'
            self.args.exp_name += f'__use_minkloc'
            self.args.exp_name += f'__{self.args.epochs}'
            self.args.exp_name += f'__{self.args.train_batch_size}'
            self.args.exp_name += f'__{self.args.image_lr}'
            self.args.exp_name += f'__{self.args.cloud_lr}'
            self.args.exp_name += f'__{self.args.useminkloc_final_embedding_type}'
        else:
            self.args.exp_name += f'__{self.args.dataset}'
            self.args.exp_name += f'__{self.args.image_fe}'
            self.args.exp_name += f'__{self.args.epochs}'
            self.args.exp_name += f'__{self.args.train_batch_size}'
            self.args.exp_name += f'__{self.args.image_lr}'
            self.args.exp_name += f'__{self.args.cloud_lr}'
            self.args.exp_name += f'__{self.args.num_other_stage_blocks}'
            self.args.exp_name += f'__{self.args.num_stage3_blocks}'
            self.args.exp_name += f'__{self.args.final_embedding_type}'
            self.args.exp_name += f'__ffblocal{self.args.use_ffblocal}'
            self.args.exp_name += f'__{self.args.ffblocal_windowsize}'
            self.args.exp_name += f'__before{self.args.usemeconv1x1_beforesptr}'
            self.args.exp_name += f'__after{self.args.usemeconv1x1_aftersptr}'


            self.args.exp_name += f'__imageswin{self.args.image_useswin}'


            self.args.exp_name += f'__useattn2{self.args.use_attn2}'
            self.args.exp_name += f'__useattn3{self.args.use_attn3}'
            self.args.exp_name += f'__useattnlocalffb{self.args.use_attninlocalffb}'
            self.args.exp_name += f'__fusion{self.args.gatt_fusion_block_type}'
            self.args.exp_name += f'__fusion2{self.args.gatt_fusion_block_type2}'
            self.args.exp_name += f'__fusion3{self.args.gatt_fusion_block_type3}'





        expr_dir = os.path.join(self.args.logdir, self.args.exp_name)
        self.args.results_dir = os.path.join(expr_dir, self.args.results_dir)
        self.args.models_dir = os.path.join(expr_dir, self.args.models_dir)
        self.args.runs_dir = os.path.join(expr_dir, self.args.runs_dir)
        mkdirs([self.args.logdir, expr_dir, self.args.runs_dir, self.args.models_dir, self.args.results_dir])

        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)


        return self.args


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)




if __name__ == '__main__':
    args = Options().parse()
