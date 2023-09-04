import numpy as np
from pycse import nlinfit

def my_nlinfit(fun, p0, xdata, ydata, **kwargs):
    def residual(p, x, y):
        return y - fun(p, x)

    def jacobian(p, x, y):
        h = 1e-6
        jac = np.zeros((len(y), len(p)))
        for i in range(len(p)):
            dp = np.zeros(len(p))
            dp[i] = h
            jac[:, i] = (residual(p + dp, x, y) - residual(p - dp, x, y)) / (2 * h)
        return jac

    result = nlinfit(residual, p0, xdata, ydata, Dfun=jacobian, **kwargs)
    params = result[0]
    cov_matrix = result[1]
    residuals = residual(params, xdata, ydata)
    jacobian_matrix = jacobian(params, xdata, ydata)
    return params, cov_matrix, residuals, jacobian_matrix

# 使用示例
def model(params, x):
    a, b, c = params
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 1, 10)
ydata = 2 * np.exp(-3 * xdata) + 0.5

p0 = [1, 1, 1]
params, cov_matrix, residuals, jacobian_matrix = my_nlinfit(model, p0, xdata, ydata)

print("Params:", params)
print("Covariance Matrix:", cov_matrix)
print("Residuals:", residuals)
print("Jacobian Matrix:", jacobian_matrix)