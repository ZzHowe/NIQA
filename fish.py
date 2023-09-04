import numpy as np
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import matlab
import matlab.engine

engine = matlab.engine.start_matlab()

def fish(org_img):

    org_img = org_img.astype(np.float64)
    lvl = 3
    org_img = org_img.tolist()
    bands = engine.dwt_cdf97(org_img, lvl)

    alpha = np.array([4, 2, 1])

    ss3 = ssq(bands[2][2])
    ss2 = ssq(bands[1][2])
    ss1 = ssq(bands[0][2])

    ss = np.array([ss1, ss2, ss3])

    dst = np.max(np.sum(ss * alpha))

    return dst


def ssq(band):
    alpha = 0.8

    lh_img = band[0] ** 2
    hl_img = band[1] ** 2
    hh_img = band[2] ** 2

    E_lh = np.log10(1 + np.mean(lh_img))
    E_hl = np.log10(1 + np.mean(hl_img))
    E_hh = np.log10(1 + np.mean(hh_img))

    ss = alpha * E_hh + (1 - alpha) * (E_lh + E_hl) / 2

    return ss