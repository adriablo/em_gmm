#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import em_gmm
import lib_test


"""
Housing data, clusters are poorly defined/separated
"""


def main():
    # print("random state: {}".format(np.random.get_state()))
    data = np.genfromtxt("kc_house_data.csv", delimiter=",", usecols=[2, 5, 17],
                         skip_header=1, max_rows=1000)
    print (data.shape)
    N = data.shape[0]  # number of data points (vectors)
    sample = data[np.random.randint(0, N, 50), :]
    lib_test.cluster(sample, 4, 1e-6)


if __name__ == '__main__':
    main()
