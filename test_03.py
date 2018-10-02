#!/usr/bin/env python3

"""
Synthetic data from Gaussian clusters;  clusters are clearly defined
https://cs.joensuu.fi/sipu/datasets/
"""

import numpy as np
import em_gmm
import lib_test


def main():
    D = 3
    data = np.genfromtxt("dim3.csv", delimiter=",", usecols=[0, 1, 2],
                         skip_header=0, max_rows=500)
    print (data.shape)
    N = data.shape[0]  # number of data points (vectors)
    sample = data[np.random.randint(0, N, 150), :]
    lib_test.cluster(sample, 3, 1e-4)
    # Harder:
    # D = 7
    # data = np.genfromtxt("dim7.csv", delimiter=",", usecols=None,
    #                      skip_header=0, max_rows=2000)
    # print (data.shape)
    # N = data.shape[0]  # number of data points (vectors)
    # sample = data[np.random.randint(0, N, 150), :]
    # lib_test.cluster(sample, 4, 1e-6)


if __name__ == '__main__':
    main()
