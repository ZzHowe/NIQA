import numpy as np

def DCTBasis(patchSize):
    N = patchSize
    W = np.zeros((N**2, N**2))
    k = 0
    omega = np.zeros((N**2,1))

    for p in range(N):
        for q in range(N):
            if p == 0:
                ap = 1 / np.sqrt(N)
            else:
                ap = np.sqrt(2 / N)
            if q == 0:
                aq = 1 / np.sqrt(N)
            else:
                aq = np.sqrt(2 / N)

            # 生成 (p,q) 滤波器
            w = np.zeros((N, N))
            for m in range(N):
                for n in range(N):
                    w[m, n] = ap * aq * np.cos(np.pi * (2 * m + 1) * p / (2 * N)) * np.cos(np.pi * (2 * n + 1) * q / (2 * N))

            W[k, :] = np.array(w).reshape(1,N * N,order='F')
            omega[k] = np.sqrt(p**2 + q**2)
            k += 1


    # 按空间频率总和排序
    I = np.argsort(omega,axis = 0)
    omega = omega[I]

    W = W[I, :].reshape((64,64))
    W = np.array(W)

    return W, omega