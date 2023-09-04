import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color.colorconv import _prepare_colorarray, rgb2hsv
import cv2
import fish
import matlab.engine
eng = matlab.engine.start_matlab()


img = cv2.imread("3.png")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = img.tolist()

print(eng.fish(matlab.double(img)))
# print("score",sharpness_f)