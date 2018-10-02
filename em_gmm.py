#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.spatial
import itertools
import operator
import matplotlib.pyplot as plt


"""
See:  ExpectationMaximization.pdf ; Expectation Maximization Tutorial by Avi Kak
"""


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return normalized_features


def pdf_mv_gaussian(x, mean, covar):
    """
    Compute density (pdf) of multivariate Gaussian in point x
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case
    Note:
    multivariate_normal from SciPy converges much slower (larger # if iterations), and the solution is worse!
    eg
    from scipy.stats import multivariate_normal
    multivariate_normal.pdf(x, mean, cov)
    """
    k = x.shape[0]
    # print "k= {}".format(k)
    # print "mean= {}".format(mean)
    # print "covar= {}".format(covar)
    det_covar = np.linalg.det(covar)
    # print "det_covar={0}".format(det_covar)
    term_1 = 1 / np.sqrt(np.power(2*np.pi, k) * det_covar)
    # print "term_1={0}".format(term_1)
    #
    inv_covar = np.linalg.inv(covar)
    u = x - mean
    exp_term = (-1.0/2) * np.dot(np.dot(u.transpose(), inv_covar), u)
    # print "exp_term={0}".format(exp_term)
    term_2 = np.exp(exp_term)
    #
    pdf = term_1 * term_2
    return pdf


class EmGmm:
    """

    """
    def __init__(self, k, data):
        self.K = k  # number of clusters (labels)
        self.N = data.shape[0]  # number of data points (vectors)
        self.D = data.shape[1]  # dimension of data vectors
        self.X = data  # data matrix (N row vectors of dimension D)

        # model parameters: K Gaussians of dimension D, each with mean vector and covariance matrix
        v_max = np.amax(data, 0)
        # print "v_max: {0}".format(v_max)
        v_min = np.amin(data, 0)
        # print "v_min: {0}".format(v_min)
        # start with random means, scaled to the data
        self.means = np.random.rand(self.K, self.D) * (v_max - v_min) + v_min
        # start with diagonal covariance matrices
        self.covariances = np.zeros((self.K, self.D, self.D))
        for l in range(self.K):
            self.covariances[l, :, :] = np.identity(self.D)
        # cluster priors
        self.a = np.ones(self.K) * (1.0 / self.K)  # equal prob. for each cluster
        # posteriors P[point, label]
        self.posteriors = None  # np.zeros((self.N, self.K))

    def compute_posteriors(self):
        """
        Compute the posterior probability of every label in each data point (vector)
        :return:
        First compute the numerators, then sum up the rows to get the per-cluster denominators
        #
        Eqn (65)
        """
        numerators = np.zeros((self.K, self.N))
        # print self.N
        for i in range(self.N):
            # print "i: {}".format(i)
            for l in range(self.K):
                # print "l: {}".format(l)
                p_x = pdf_mv_gaussian(self.X[i, :], self.means[l, :], self.covariances[l, :])  # likelihood
                # print "p_x: {}".format(p_x)
                numerators[l, i] = p_x * self.a[l]
        # now compute the denominators
        # the denominator is the sum of the numerators over labels, on each datapoint
        denominators = np.sum(numerators, 0)
        self.posteriors = (numerators / denominators).transpose()  # (N, K)

    def update_priors(self):
        self.a = (1.0 / self.N) * np.sum(self.posteriors, 0)

    def update_means(self):
        """
        Eqn (53), (54)
        :return:
        """
        numerators = np.zeros((self.D, self.K))
        for l in range(self.K):
            for i in range(self.N):
                numerators[:, l] += self.X[i, :] * self.posteriors[i, l]
        denominators = np.sum(self.posteriors, 0)
        self.means = (numerators / denominators).transpose()  # (K, D)

    def update_covariances(self):
        """
        Equations (63), (64)
        :return:
        """
        numerators = np.zeros((self.K, self.D, self.D))
        for l in range(self.K):
            for i in range(self.N):
                d = np.matrix((self.X[i, :] - self.means[l, :]), )
                # print "d: {}".format(d)
                # quadratic form
                Q = d.transpose().dot(d)
                # print "Q: {}".format(Q)
                numerators[l, :] += Q * self.posteriors[i, l]
        denominators = np.sum(self.posteriors, 0)  # (K)
        self.covariances = (numerators.transpose() / denominators).transpose()  # (K, D, D)

    def fit(self, max_steps=1000, min_error=1e-6):
        n = 1
        e = 1 + min_error
        old_means = self.means
        while n <= max_steps and e > min_error:
            self.compute_posteriors()
            self.update_priors()
            self.update_means()
            self.update_covariances()
            e = np.linalg.norm(self.means - old_means)
            old_means = self.means
            n += 1
            print("n= {}, e= {}".format(n, e))
        pass


def main():
    print('Gaussian mixture clustering library')


if __name__ == '__main__':
    main()
