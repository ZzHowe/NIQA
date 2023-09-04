import numpy as np
from scipy.stats import kurtosis
from DCTBasis import DCTBasis
from fitShapeAndSDUsingKurts import fitShapeAndSDUsingKurts
from scipy.signal import convolve2d
from scipy import stats

def estimateNoiseSDUsingKurts(noiseI, patchSize):
    N = patchSize**2

    # 创建DCT基础滤波器
    W,omega = DCTBasis(patchSize)

    # 从图像中去除直流成分
    noiseI = noiseI - np.mean(noiseI)
    noiseI[noiseI < 0] = 0

    # 收集图像的统计信息
    noiseVars = np.zeros(N)
    noiseKurts = np.zeros(N)


    # for i in range(len(W[0])):
    #     W[i,:] = np.array(W[i])

    for i in range(1, N):
        temp = convolve2d(noiseI, W[i, :].reshape(patchSize, patchSize,order='F'), 'valid')
        noiseVars[i] = np.var(temp)
        noiseKurts[i] = kurtosis(temp.reshape(-1),fisher = False)



    # 找到最能解释统计信息的噪声标准差和峰度
    noiseSD, estimatedKurt = fitShapeAndSDUsingKurts(noiseVars, noiseKurts)
    noiseSD = abs(noiseSD)


    return noiseSD