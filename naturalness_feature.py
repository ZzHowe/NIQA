import numpy as np
from scipy.ndimage import filters
from scipy.special import gamma
import cv2

def naturalness_feature(imdist, scalenum=1):

    imdist = imdist.astype(np.float64)

    window = np.multiply(cv2.getGaussianKernel(7, 7/6), (cv2.getGaussianKernel(7, 7/6)).T)


    feat = np.array([])

    for itr_scale in range(scalenum):
        mu = filters.convolve(imdist, window, mode='constant', cval=0.0)
        mu_sq = mu ** 2
        sigma = np.sqrt(np.abs(filters.convolve(imdist ** 2, window, mode='constant', cval=0.0) - mu_sq))
        structdis = (imdist - mu) / (sigma + 1)

        alpha = estimateggdparam(structdis.flatten())[0]
        overallstd = np.std(structdis.flatten())
        feat = np.append(feat, [alpha, overallstd ** 2])

    return alpha, overallstd ** 2

def estimateggdparam(vec):
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (gamma(1. / gam) * gamma(3. / gam)) / (gamma(2. / gam) ** 2)

    sigma_sq = np.mean(vec ** 2)
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / E ** 2
    array_position = np.argmin(np.abs(rho - r_gam))
    gamparam = gam[array_position]

    return gamparam, sigma