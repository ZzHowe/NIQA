import cv2
import numpy as np

from nnid_feat import nnid_feat

img = cv2.imread('03.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

feat = nnid_feat(img)
print(feat)