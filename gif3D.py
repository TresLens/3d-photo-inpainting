import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from glob2 import glob
from skimage.metrics import structural_similarity as ssim
import os
import shutil
import subprocess
from PIL import Image
from keras_segmentation.pretrained import pspnet_101_voc12
import gc
from time import time

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

class Gif3D:
    def __init__(self) -> None:
        print('Loading segmentation model')
        t1 = time()
        self.model = pspnet_101_voc12()
        print('Segmentation model loaded in {} seconds'.format(time() - t1))

    def segmentation(self, img):
        seg = self.model.predict_segmentation(img)
        mask = cv2.inRange(seg, np.array([14]), np.array([16]))
        img_masked = cv2.bitwise_and(img, img, mask=cv2.resize(mask, img.shape[:-1][::-1]).astype(np.uint8))
        return mask, img_masked

    def run_3dgif(self, src_folder) -> str:
        gc.collect()
        video_src = glob(src_folder + '/*.mp4')[0].replace('\\', '/')
        cap = cv2.VideoCapture(video_src)
        id_file = video_src.split('/')[-1].split('_')[0]

        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if len(frames) == 0:
                frames.append(frame)
            elif self.__check_frames(frames, frame) == True:
                break
        cap.release()

        img_for_seg = cv2.cvtColor(frames[5], cv2.COLOR_RGB2BGR)

        mask, img_masked = self.segmentation(img_for_seg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in sorted(contours, key=cv2.contourArea)[::-1]:
            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [c], -1, 255, cv2.FILLED, 1)
            img_masked = cv2.bitwise_and(img_for_seg, img_for_seg, mask=cv2.resize(new_mask, img_for_seg.shape[:-1][::-1]).astype(np.uint8))

            try:
                disp = [self.__get_disp_image(img_masked, f) for f in frames]
                break
            except IndexError:
                continue

        gif_imgs = []
        min_cut, max_cut = -min(disp), max(disp)
        print('Min: ', min_cut, ' Max: ', max_cut)
        rows, cols = frames[0].shape[:-1]
        for i in range(0, len(frames)):
            M = np.float32([[1,0,-disp[i]],[0,1,0]])
            img_mvd = cv2.warpAffine(frames[i],M,(cols,rows))
            img_mvd = img_mvd[0:rows, max_cut:cols-min_cut]
            gif_imgs.append(img_mvd)
        gif_imgs += gif_imgs[1:][::-1]

        dir_name = f'{src_folder}/frames/{id_file}'
        os.makedirs(dir_name, exist_ok=True)
        [cv2.imwrite(f'{dir_name}/{i}.jpg', gif_imgs[i]) for i in range(len(gif_imgs))]
        subprocess.run(f'ffmpeg -i {dir_name}/%d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2, fps=15" -pix_fmt yuv420p {src_folder}/{id_file}.mp4', shell=True)
        # subprocess.run(f'ffmpeg -i gifs/{id_file}.mp4 -filter_complex "[0]reverse[r];[0][r]concat,loop=5:100,setpts=N/15/TB" gifs/{id_file}.mp4', shell=True)
        shutil.rmtree(dir_name)

    def __check_frames(self, frames, frame):
        for f in frames:
            if mse(f, frame) < 100:
                return True
        frames.append(frame)
        
    def __get_disp_image(self, img_masked, img):
        img1 = img_masked
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        r,c = img2.shape[:-1]
        
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des1,des2)
        
        def match_get_coords(match):
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx

            x1, y1 = kp1[img1_idx].pt
            x2, y2 = kp2[img2_idx].pt

            return (x1, y1, x2, y2)
        
        def filter_matches(match):
            _, y1, _, y2 = match_get_coords(match)
            e = abs(y1-y2)
            return e < 1

        matches = list(filter(filter_matches, matches))
        matches = sorted(matches, key=lambda m: m.distance)
        
        def get_offset(match):
            x1, _, x2, _ = match_get_coords(match)
            return x2-x1

        return int(get_offset(matches[0])) #np.mean([get_offset(m) for m in matches])

if __name__ == '__main__':
    gf = Gif3D()
    gf.run_3dgif('image')