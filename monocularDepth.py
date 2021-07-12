import sys
import os
sys.path.append(os.getcwd() + '/BoostingMonocularDepth')

from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms

# OUR
from BoostingMonocularDepth.utils import ImageandPatchs, ImageDataset, generatemask, getGF_fromintegral, calculateprocessingres, rgb2gray,\
    applyGridpatch

# MIDAS
import BoostingMonocularDepth.midas.utils
from BoostingMonocularDepth.midas.models.midas_net import MidasNet
from BoostingMonocularDepth.midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# PIX2PIX : MERGE NET
from BoostingMonocularDepth.pix2pix.options.test_options import TestOptions
from BoostingMonocularDepth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

from BoostingMonocularDepth.run import doubleestimate, generatepatchs

import time
import torch
import cv2
import numpy as np
import argparse
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class MonocularDepth:
    def __init__(self, depthNet = 0) -> None:
        self.device = torch.device("cuda")
        self.whole_size_threshold = 3000  # R_max from the paper
        self.GPU_threshold = 2000 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted
        self.pix2pixsize = 1024
        self.depthNet = depthNet

        # Load merge network
        opt = TestOptions().parse()
        self.pix2pixmodel = Pix2Pix4DepthModel(opt)
        self.pix2pixmodel.save_dir = 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel'
        self.pix2pixmodel.load_networks('latest')
        self.pix2pixmodel.eval()

        if depthNet == 0:
            self.net_receptive_field_size = 384
            self.patch_netsize = 2*self.net_receptive_field_size
        elif depthNet == 1:
            self.net_receptive_field_size = 448
            self.patch_netsize = 2*self.net_receptive_field_size

        # Decide which depth estimation network to load
        if depthNet == 0:
            midas_model_path = "BoostingMonocularDepth/midas/model.pt"
            self.midasmodel = MidasNet(midas_model_path, non_negative=True)
            self.midasmodel.to(self.device)
            self.midasmodel.eval()
        elif depthNet == 1:
            from BoostingMonocularDepth.structuredrl.models import DepthNet
            self.srlnet = DepthNet.DepthNet()
            self.srlnet = torch.nn.DataParallel(self.srlnet, device_ids=[0]).cuda()
            checkpoint = torch.load('BoostingMonocularDepth/structuredrl/model.pth.tar')
            self.srlnet.load_state_dict(checkpoint['state_dict'])
            self.srlnet.eval()

        self.mask_org = generatemask((3000, 3000))
        self.mask = self.mask_org.copy()

        self.r_threshold_value = 0.2

    def run_monocularDepth(self, src_dir, result_dir) -> None:
        torch.cuda.empty()
        dataset = ImageDataset(src_dir, 'test')
        for image_ind, images in enumerate(dataset):
            img = images.rgb_image
            input_resolution = img.shape

            scale_threshold = 3  # Allows up-scaling with a scale up to 3

            whole_image_optimal_size, patch_scale = calculateprocessingres(img, self.net_receptive_field_size,
                                                                       self.r_threshold_value, scale_threshold,
                                                                       self.whole_size_threshold)

            whole_estimate = doubleestimate(img, self.net_receptive_field_size, whole_image_optimal_size,
                                        self.pix2pixsize, self.depthNet, self.pix2pixmodel, self.GPU_threshold, midasmodel=self.midasmodel)

            self.factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / self.whole_size_threshold), 0.2)

            if img.shape[0] > img.shape[1]:
                a = 2*whole_image_optimal_size
                b = round(2*whole_image_optimal_size*img.shape[1]/img.shape[0])
            else:
                a = round(2*whole_image_optimal_size*img.shape[0]/img.shape[1])
                b = 2*whole_image_optimal_size

            img = cv2.resize(img, (round(b/self.factor), round(a/self.factor)), interpolation=cv2.INTER_CUBIC)

            base_size = self.net_receptive_field_size*2

            patchset = generatepatchs(img, base_size, self.factor)

            mergin_scale = input_resolution[0] / img.shape[0]

            imageandpatchs = ImageandPatchs(src_dir, images.name, patchset, img, mergin_scale)
            whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergin_scale),
                                                round(img.shape[0]*mergin_scale)), interpolation=cv2.INTER_CUBIC)
            imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
            imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

            for patch_ind in range(len(imageandpatchs)):
                self.__process_patch(patch_ind, imageandpatchs)
            
            path = os.path.join(result_dir, imageandpatchs.name)

            BoostingMonocularDepth.midas.utils.write_depth(path,
                                    cv2.resize(imageandpatchs.estimation_updated_image,
                                               (input_resolution[1], input_resolution[0]),
                                               interpolation=cv2.INTER_CUBIC), bits=2, colored=False)

    def __process_patch(self, patch_ind, imageandpatchs):
        patch = imageandpatchs[patch_ind] # patch object
        patch_rgb = patch['patch_rgb'] # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
        rect = patch['rect'] # patch size and location
        patch_id = patch['id'] # patch ID
        org_size = patch_whole_estimate_base.shape # the original size from the unscaled input

        # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
        # field size of the network for patches to accelerate the process.
        patch_estimation = doubleestimate(patch_rgb, self.net_receptive_field_size, self.patch_netsize,
                                        self.pix2pixsize, self.depthNet, self.pix2pixmodel, self.GPU_threshold, midasmodel=self.midasmodel)

        patch_estimation = cv2.resize(patch_estimation, (self.pix2pixsize, self.pix2pixsize),
                                    interpolation=cv2.INTER_CUBIC)

        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (self.pix2pixsize, self.pix2pixsize),
                                            interpolation=cv2.INTER_CUBIC)

        # Merging the patch estimation into the base estimate using our merge network:
        # We feed the patch estimation and the same region from the updated base estimate to the merge network
        # to generate the target estimate for the corresponding region.
        self.pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        self.pix2pixmodel.test()
        visuals = self.pix2pixmodel.get_current_visuals()

        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped+1)/2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        mapped = prediction_mapped

        # We use a simple linear polynomial to make sure the result of the merge network would match the values of
        # base estimate
        p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

        merged = cv2.resize(merged, (org_size[1],org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
        # and resize it to our needed size while merging the patches.
        if self.mask.shape != org_size:
            mask = cv2.resize(self.mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        # Update the whole estimation:
        # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
        # blending at the boundaries of the patch region.
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

if __name__ == '__main__':
    md = MonocularDepth(0)
    md.run_monocularDepth('image/image', 'image/frames')