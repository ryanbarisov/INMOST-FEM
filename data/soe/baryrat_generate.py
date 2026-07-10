import numpy as np
import matplotlib.pyplot as plt
import baryrat
import sys

# see https://arxiv.org/pdf/2111.05121

delt = 0

def generate_coefs(alph, nexp, intvl):
    def LaplK(s):
	    return np.power(s+delt, alph-1)
    r = baryrat.brasil(LaplK, intvl, nexp)
    b, a = r.polres()
    b *= -1
    gamma2 = r.gain()
    return gamma2, a, b


if __name__ == "__main__":
    nexp = 4
    #n = 99
    #alphas = np.linspace(0.01, 0.99, n)
    n=9
    alphas=np.linspace(0.1,0.9,n)
    intvl = [1e-3,2e3]
    if len(sys.argv) > 1:
        nexp = int(sys.argv[1])
    if len(sys.argv) > 2:
        intvl[1] = float(sys.argv[2])
    print(f'generate sum of exponent approximation (nexp = {nexp})\nalpha from {np.min(alphas)} to {np.max(alphas)} on interval {intvl}')
    G, A, B = [], [], []
    for i in range(len(alphas)):
        gamma2, a, b = generate_coefs(alphas[i], nexp, intvl)
        G.append(gamma2)
        A.append(a)
        B.append(b)
    with open(f'coefs_nexp{nexp:02d}_T{int(intvl[1]):05d}.csv', 'w') as f:
        f.write(f'{nexp}\n')
        for i in range(n):
            f.write(f'{alphas[i]:.2e};{G[i]:.9e}')
            for j in range(nexp):
                f.write(f';{A[i][j]:.9e}')
            for j in range(nexp):
                f.write(f';{B[i][j]:.9e}')
            f.write('\n')
