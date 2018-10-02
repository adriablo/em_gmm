#!/usr/bin/env python3

"""
library for clustering tests
"""

import numpy as np
import matplotlib.pyplot as plt
import em_gmm


def plot(data, i, j, centers, labels, colors, col_vec):
    x, y = data[:, i], data[:, j]
    plt.scatter(x, y, c=colors, s=30, edgecolor="gray")
    plt.scatter(centers[:, i], centers[:, j], marker="D", c=col_vec, s=200)
    plt.xlim(x.min()*.9, x.max()*1.1)
    plt.ylim(y.min()*.9, y.max()*1.1)
    plt.show()


def cluster(data, n_clust, min_error):
    """
    Cluster in D dimensions then plot each 2D pair, eg (x1, x2), (x1, x3), (x2, x3), ..., (x3, x11), ...
    :param data:
    :param n_clust:
    :param min_error:
    :return:
    """
    data = em_gmm.normalize_features(data)
    # print data
    e = em_gmm.EmGmm(n_clust, data)
    e.fit(min_error=min_error)
    # print e.means
    labels = np.argmax(e.posteriors, 1)
    # print labels
    centers = e.means
    print('centers: {}'.format(centers))
    D = data.shape[1]  # dimension of data vectors
    #
    col_vec = ["red", "blue", "green", "purple", "black", "yellow", "cyan", "magenta", "pink"]
    colors = [col_vec[x] for x in labels]
    # plot each pair of dimensions ie x[i] vs x[j]
    for i in range(D):
        for j in range(i + 1, D):
            print(i, j)
            plot(data, i, j, centers, labels, colors, col_vec)


if __name__ == '__main__':
    assert False
