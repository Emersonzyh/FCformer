import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
from PIL import Image
# /////////////// Distortion Helpers ///////////////
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
input_dir = "./r_h_gb/"
# output_dir = "./r_h_gb_gn/"
output_dir = "./degradations/"
imgs = os.listdir(input_dir)
input_img = "./train_clean/50.jpg"
# for i in imgs:
img = convert_img(cv2.imread(input_img))
res = shot_noise(img, 3)
# cv2.imwrite(output_dir + i.split('.')[0] + "_gn.png", res) # gn, sn
cv2.imwrite(output_dir + "50_sn.png", res) # gn, sn