import numpy as np
from scipy.optimize import curve_fit
from pycse import nlinfit
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from scipy.stats import t

def predict_func(x, params):
    return logistic(x, *params)

def logistic(X,bayta1,bayta2,bayta3,bayta4,bayta5):

    logisticPart = 0.5 - 1 / (1 + np.exp(bayta2 * (X - bayta3)))

    yhat = bayta1 * logisticPart + bayta4 * X + bayta5

    return yhat

def nlpredci(func, xdata, params, residuals, jac):
    alpha = 0.05  # 置信水平
    dof = len(xdata) - len(params)  # 自由度
    tval = t.ppf(1 - alpha / 2, dof)  # t分布值
    popt = params  # 参数估计值

    ypred = func(xdata, *params)  # 预测值
    se = np.sqrt(np.sum(residuals ** 2) / dof)  # 标准误差

    return ypred


def verify_performance(mos, predict_mos):
    predict_mos = predict_mos.flatten()
    mos = mos.flatten()
    print('predict_mos.shape')
    print(predict_mos.shape)
    print('mos.shape')
    print(mos.shape)

    beta = [10, 0, np.mean(predict_mos), 0.1, 0.1]

    # Fitting a curve using the data
    bayta,pcov = curve_fit(logistic, predict_mos, mos, p0 = beta)
    residuals = mos - logistic(predict_mos, *bayta)

    # 计算雅可比矩阵
    jac = np.zeros((len(predict_mos), len(bayta)))

    ypre = logistic(predict_mos, *bayta)


    rmse = np.sqrt(np.sum((ypre - mos) ** 2) / len(mos))  # Root Mean Squared Error
    plcc, _ = pearsonr(mos, ypre)  # Pearson Linear Correlation Coefficient
    srocc, _ = spearmanr(mos, predict_mos)  # Spearman Rank Correlation Coefficient
    krocc, _ = kendalltau(mos, predict_mos)  # Kendall Rank Correlation Coefficient
    #
    return srocc, krocc, plcc, rmse