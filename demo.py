import cv2
from nnid_feat import  nnid_feat
from contrast_feat import contrast_feat
from naturalness_feature import naturalness_feature
from fish import fish
img = cv2.imread("3.png")


# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

alpha, overallstd = naturalness_feature(img)
print(alpha)
print(overallstd)