import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color.colorconv import _prepare_colorarray, rgb2hsv
import cv2
import fish_bb

def rgb2gray(rgb, *, channel_axis=-1):
    rgb = _prepare_colorarray(rgb)
    coeffs = np.array([0.2989, 0.5870, 0.1140], dtype=rgb.dtype)
    return rgb @ coeffs


img = cv2.imread("3.png")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = rgb2gray(img)
score,map = fish_bb.fish_bb("1.png")