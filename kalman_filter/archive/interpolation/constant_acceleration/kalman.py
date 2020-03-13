import numpy as np
import pandas as pd
import math
from math import pi
from numpy import dot, sum, tile, linalg, exp, log, shape, ndim
from numpy.linalg import inv

"""
    The algorythm is heavily based on the following paper: https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf.
    See the interneto read more about how the filter works.

    X : The mean state estimate of the previous step ( k −1).
    P : The state covariance of previous step ( k −1).
    A : The transition n n × matrix.
    Q : The process noise covariance matrix.
"""


def kf_predict(X_minus, P_minus, A, Q):
    """

    Args:
        X_minus:
        P_minus:
        A:
        Q:

    Returns:

    """
    # Project the state ahead
    x = A * X_minus

    # Project the error covariance ahead
    P = A * P_minus * A.T + Q

    return x, P


"""
    K : the Kalman Gain matrix
    IM : the Mean of predictive distribution of Y
    IS : the Covariance or predictive mean of Y
    LH : the Predictive probability (likelihood) of measurement which is computed using the Python function gauss_pdf.
"""


def kf_update(X_predicted, P_predicted, I, H, lng, lat, a_east, a_north, sigma_pos, sigma_acc):
    """

    Args:
        X_predicted:
        P_predicted:
        I:
        H:
        lng:
        lat:
        a_east:
        a_north:
        sigma_pos:
        sigma_acc:

    Returns:

    """
    R = np.matrix([[sigma_pos, 0.0, 0.0, 0.0],
                   [0.0, sigma_pos, 0.0, 0.0],
                   [0.0, 0.0, sigma_acc, 0.0],
                   [0.0, 0.0, 0.0, sigma_acc]])

    # Compute the Kalman Gain
    S = H * P_predicted * H.T + R
    K = (P_predicted * H.T) * np.linalg.pinv(S)

    # Update the estimate via z
    Z = np.array([lng, lat, a_east, a_north]).reshape(H.shape[0], 1)
    IM = H * X_predicted
    y = Z - IM  # Innovation or Residual
    x = X_predicted + (K * y)

    # Update the error covariance
    P = (I - (K * H)) * P_predicted
    #LH = gauss_pdf(y, IM, S)
    rows = []
    cols = []

    for row in S:
        #print(np.array(row)[0])
        for col in np.array(row)[0]:
            #print('c',col)
            if col > 1e19:
                col = 1e18
            cols.append(col)
        # cols[3] *=1.000011
        # cols[2] *= 1.000012
        # cols[0] *= 1.000013
        # cols[1] *= 1.000014

        rows.append(np.array(cols))
        cols = []

    S = np.array(rows)
    print(type(S))
    print(S)
    eps = np.dot(y.T, S).dot(y)

    print('EPSILON', eps.A[0])
    print(type(eps.A[0][0]))
    if(eps.A[0][0]) > 10000:
        print('bigger', eps)

    # print('YAY', LH)

    return x, P, Z, K, eps.A[0][0]


def gauss_pdf(X, M, S):
    """
    https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions
    https://en.wikipedia.org/wiki/Probability_density_function
    Args:
        self:
        X:
        M:
        S:

    Returns:

    """
    if ndim(M) == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (np.dot(np.inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1]) - M
        E = 0.5 * sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    else:
        DX = X - M
        E = 0.5 * np.dot(DX.T, np.dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
        P = exp(-E)
    return (P[0], E[0])
