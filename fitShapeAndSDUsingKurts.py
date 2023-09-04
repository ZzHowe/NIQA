import numpy as np
from scipy.optimize import fmin

def fitShapeAndSDUsingKurts(noiseVars, noiseKurts):
    myminfun = lambda guesses: sum([(kurtFunc(guesses[0], guesses[1], i, noiseVars) - noiseKurts[i]) ** 2 for i in range(1, len(noiseVars))])
        # 自定义的目标函数，在这里进行计算
        # 返回目标函数的值

    # 设置优化过程的初始猜测
    initial_guess = [np.min(noiseKurts[1:]), np.sqrt(np.min(noiseVars[1:]))]

    # 设置优化过程的最大函数评估次数和最大迭代次数
    #options = {'maxfun': 100000, 'maxiter': 100000}

    # 调用优化函数进行最小化优化
    result = fmin(myminfun, initial_guess, maxiter=100000,maxfun = 100000)

    # 从结果中获取优化后的变量
    x = result
    noiseSD = x[1]
    kurt = x[0]

    # 检查退出标志
    exitflag = 0 if np.isnan(noiseSD) or np.isnan(kurt) else 1

    # 如果退出标志为 0，将变量设置为 NaN
    if exitflag == 0:
        noiseSD = np.nan
        kurt = np.nan

    return noiseSD, kurt



def kurtFunc(kurt, sd, i, noiseVars):
    f = lambda x: (kurt - 3) / ((1 + x) ** 2) + 3
    return f((sd ** 2) / (noiseVars[i] - sd ** 2))