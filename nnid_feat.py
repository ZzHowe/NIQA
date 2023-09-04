import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color.colorconv import _prepare_colorarray, rgb2hsv
from fish import fish
import cv2
from estimateNoiseSDUsingKurts import estimateNoiseSDUsingKurts
from contrast_feat import contrast_feat
from naturalness_feature import naturalness_feature
import matlab.engine
engine = matlab.engine.start_matlab()

def nnid_feat(img):
    color_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    #luminance 求亮度
    overall_lumi_f = np.mean(img)
    # lumi_tmp = uniform_filter(img, size=(64,64), mode='constant')
    # luminance_var = np.std(lumi_tmp)

    # saturation 求饱和度 HSV
    hsv_img = rgb2hsv(color_img)
    saturation_f = np.mean(hsv_img[:, :, 1])

    #Sharpness 求锐度
    img_fish = img.tolist()
    sharpness_f = engine.fish(matlab.double(img_fish))

    #noiseness 求噪声等级
    noiseness_f = estimateNoiseSDUsingKurts(img, 8)

    #contrast histogram equlism 求对比度
    contrast_f = contrast_feat(color_img)

    #naturalness
    alpha, overallstd = naturalness_feature(img)

    feat = [overall_lumi_f, saturation_f, sharpness_f, noiseness_f, contrast_f, alpha, overallstd]
    return feat