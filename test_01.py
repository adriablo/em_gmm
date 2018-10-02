#!/usr/bin/env python3

"""
Synthetic data from Gaussian clusters;  clusters are clearly defined
https://cs.joensuu.fi/sipu/datasets/
"""

import numpy as np
import em_gmm
import lib_test


def main():
    """
    Converges relatively quickly to a good solution
    :return:
    """
    data = np.genfromtxt("s1.csv", delimiter=",", usecols=[0, 1],
                         skip_header=0, max_rows=1000)
    print (data.shape)
    lib_test.cluster(data, 4, 1e-7)


if __name__ == '__main__':
    main()
