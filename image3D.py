import numpy as np
import argparse
import os
from functools import partial
import vispy
import scipy.misc as misc
import yaml
import time
from mesh import write_ply, read_ply, output_3d_photo
from utils3D import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

class Image3D:
    def __init__(self, config) -> None:
        self.config = yaml.load(open(config, 'r'))
        vispy.use(app='pyqt5')

        if isinstance(self.config['gpu_ids'], int) and (self.config['gpu_ids'] >= 0):
            self.device = self.config['gpu_ids']
        else:
            self.device = "cpu"

        print(f"Loading edge model at {time.time()}")
        self.depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        self.depth_edge_weight = torch.load(self.config['depth_edge_model_ckpt'],
                                       map_location=torch.device(self.device))
        self.depth_edge_model.load_state_dict(self.depth_edge_weight)
        self.depth_edge_model = self.depth_edge_model.to(self.device)
        self.depth_edge_model.eval()

        print(f"Loading depth model at {time.time()}")
        self.depth_feat_model = Inpaint_Depth_Net()
        self.depth_feat_weight = torch.load(self.config['depth_feat_model_ckpt'],
                                       map_location=torch.device(self.device))
        self.depth_feat_model.load_state_dict(self.depth_feat_weight, strict=True)
        self.depth_feat_model = self.depth_feat_model.to(self.device)
        self.depth_feat_model.eval()
        self.depth_feat_model = self.depth_feat_model.to(self.device)

        print(f"Loading rgb model at {time.time()}")
        self.rgb_model = Inpaint_Color_Net()
        self.rgb_feat_weight = torch.load(self.config['rgb_feat_model_ckpt'],
                                     map_location=torch.device(self.device))
        self.rgb_model.load_state_dict(self.rgb_feat_weight)
        self.rgb_model.eval()
        self.rgb_model = self.rgb_model.to(self.device)
        self.graph = None


    def run_3dimage(self, src) -> str:
        src_folder = src + '/image'
        self.config['mesh_folder'] = src
        self.config['video_folder'] = src
        self.config['depth_folder'] = src + '/depth'

        sample_list = get_MiDaS_samples(src_folder, self.config['depth_folder'], self.config, self.config['specific'])
        normal_canvas, all_canvas = None, None

        for idx in range(len(sample_list)):
            depth = None
            sample = sample_list[idx]
            print("Current Source ==> ", sample['src_pair_name'])
            mesh_fi = os.path.join(self.config['mesh_folder'], sample['src_pair_name'] +'.ply')
            image = imageio.imread(sample['ref_img_fi'])
            
            self.config['output_h'], self.config['output_w'] = imageio.imread(sample['depth_fi'], as_gray=True).shape[:2]
            
            frac = self.config['longer_side_len'] / max(self.config['output_h'], self.config['output_w'])
            self.config['output_h'], self.config['output_w'] = int(self.config['output_h'] * frac), int(self.config['output_w'] * frac)
            self.config['original_h'], self.config['original_w'] = self.config['output_h'], self.config['output_w']
            if image.ndim == 2:
                image = image[..., None].repeat(3, -1)
            if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
                self.config['gray_image'] = True
            else:
                self.config['gray_image'] = False
            image = cv2.resize(image, (self.config['output_w'], self.config['output_h']), interpolation=cv2.INTER_AREA)
            depth = read_MiDaS_depth(sample['depth_fi'], 3.0, self.config['output_h'], self.config['output_w'])
            mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
            if not(self.config['load_ply'] is True and os.path.exists(mesh_fi)):
                _, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), self.config, num_iter=self.config['sparse_iter'], spdb=False)
                depth = vis_depths[-1]
                torch.cuda.empty_cache()
                print("Start Running 3D_Photo ...")
        
                print(f"Writing depth ply (and basically doing everything) at {time.time()}")
                rt_info = write_ply(image,
                                    depth,
                                    sample['int_mtx'],
                                    mesh_fi,
                                    self.config,
                                    self.rgb_model,
                                    self.depth_edge_model,
                                    self.depth_edge_model,
                                    self.depth_feat_model)

                if rt_info is False:
                    continue
                
                torch.cuda.empty_cache()
            if self.config['save_ply'] is True or self.config['load_ply'] is True:
                verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
            else:
                verts, colors, faces, Height, Width, hFov, vFov = rt_info

            print(f"Making video at {time.time()}")
            videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
            top = (self.config.get('original_h') // 2 - sample['int_mtx'][1, 2] * self.config['output_h'])
            left = (self.config.get('original_w') // 2 - sample['int_mtx'][0, 2] * self.config['output_w'])
            down, right = top + self.config['output_h'], left + self.config['output_w']
            border = [int(xx) for xx in [top, down, left, right]]
            normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                                copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(self.config['video_folder']),
                                image.copy(), copy.deepcopy(sample['int_mtx']), self.config, image,
                                videos_poses, video_basename, self.config.get('original_h'), self.config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                                mean_loc_depth=mean_loc_depth)
