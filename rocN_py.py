import numpy as np


def rocN(x, y, N=100):
    x = x.flatten("F")
    y = y.flatten("F")
    zlo = min(min(x), min(y))
    zhi = max(max(x), max(y))
    z = np.linspace(zlo, zhi, N)
    fa = np.zeros((1, N))
    hit = np.zeros((1, N))
    for i in range(N):
        fa[N - i] = sum(y > z[i])
        hit[N - i] = sum(x > z[i])

    fa = fa / y.shape[1]
    hit = hit / x.shape[1]
    a = np.trapz(y=hit, x=fa)
    return a
