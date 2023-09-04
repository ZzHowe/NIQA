import numpy as np
from scipy.ndimage import imread
from skimage.color import rgb2gray
from dwt_cdf97 import dwt_cdf97


def fish_bb(filename):
    # 读取图像
    org_img = imread(filename)

    # 如果图像是彩色图像，则转换为灰度图像
    if org_img.ndim == 3:
        org_img = rgb2gray(org_img)
    else:
        org_img = org_img.astype(float)

    # 设置参数
    lvl = 3  # DWT层数
    r, c = org_img.shape  # 图像的行数和列数
    blk = 4  # 块大小

    # 进行DWT变换
    bands = dwt_cdf97(org_img, lvl)
    print(bands)

    # 创建子带数组
    subbands = [[None] * 3 for _ in range(3)]
    r0 = (r // (2 * blk)) - 1
    c0 = (c // (2 * blk)) - 1

    # 计算鱼眼边界框
    dst = np.zeros((r0, c0))

    for idx in range(r0):
        for jdx in range(c0):
            for m in range(3):
                for n in range(3):
                    subbands[m][n] = bands[m][n][
                                     1 + (idx - 1) * blk // (2 ** m): (idx + 1) * blk // (2 ** m),
                                     1 + (jdx - 1) * blk // (2 ** m): (jdx + 1) * blk // (2 ** m)
                                     ]
            dst[idx, jdx] = map_index(subbands)

    # 计算结果
    tmp = np.sort(dst.ravel(), axis=None)[::-1]
    lth = len(tmp)
    res = np.sqrt(np.mean(tmp[:max(1, int(lth / 100))].astype(float) ** 2))

    return res, dst


def ssq(bands):
    # 计算子带的能量
    alpha = 0.8

    lh_img = bands[0] ** 2
    hl_img = bands[1] ** 2
    hh_img = bands[2] ** 2

    E_lh = np.log10(1 + np.mean(lh_img))
    E_hl = np.log10(1 + np.mean(hl_img))
    E_hh = np.log10(1 + np.mean(hh_img))

    ss = alpha * E_hh + (1 - alpha) * (E_lh + E_hl) / 2

    return ss


def map_index(bands):
    # 计算子带的指数映射
    alpha = [4, 2, 1]

    ss3 = ssq(bands[2])
    ss2 = ssq(bands[1])
    ss1 = ssq(bands[0])

    ss = [ss1, ss2, ss3]

    dst = np.sum(np.multiply(ss, alpha))

    return dst