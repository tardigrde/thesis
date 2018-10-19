import numpy as np
import pandas as pd
import math
from math import pi
from numpy import dot, sum, tile, linalg, exp, log
from numpy.linalg import inv

class Kalman:
    def __init__(self):
        pass
    """
        The algorythm is heavily based on the following paper: https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf.
        See the interneto read more about how the filter works.

        X : The mean state estimate of the previous step ( k −1).
        P : The state covariance of previous step ( k −1).
        F : The transition n n × matrix.
        Q : The process noise covariance matrix.
        B : The input effect matrix.
        U : The control input.
    """
        
    def kf_predict(self, X_minus, P_minus, F, Q, B, U): #the parameter X is the old X, and the returned value is the new value
        fx = np.dot(F, X_minus)
        bu = np.dot(B, U)
        X = X_minus + bu 
        P = np.dot(F, np.dot(P_minus, F.T)) + Q
        return {'X': X, 'P': P}
    """
    def gauss_pdf(self, X, M, S):
        if M.shape()[1] == 1:
            DX = X - tile(M, X.shape()[1])
            E = 0.5 * sum(DX * (np.dot(np.inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
            P = exp(-E)
        elif X.shape()[1] == 1:
            DX = tile(X, M.shape()[1])- M
            E = 0.5 * sum(DX * (np.dot(inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
            P = exp(-E)
        else:
            DX = X-M
            E = 0.5 * np.dot(DX.T, np.dot(inv(S), DX))
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(linalg.det(S))
            P = exp(-E)
        return (P[0],E[0])
    """
    """
        K : the Kalman Gain matrix
        IM : the Mean of predictive distribution of Y
        IS : the Covariance or predictive mean of Y
        LH : the Predictive probability (likelihood) of measurement which is computed using the Python function gauss_pdf.
    """

    def kf_update(self, X_minus, P_minus, Y, H, R):
        print(X_minus)
        print(P_minus)
        print(H)
        print(R)
        IM = np.dot(H, X_minus)
        print(Y)
        print(IM)
        Z = Y - IM
        IS = R + np.dot(H, np.dot(P_minus, H.T))
        K = np.dot(P_minus, np.dot(H.T, inv(IS)))
        kz = np.dot(K,Z)
        X = X_minus + kz
        P = P_minus - np.dot(K, np.dot(IS, K.T))
        #LH = self.gauss_pdf(Y, IM, IS)
        #print('update')
        return {'X': X,'P': P,'K': K,'IM': IM,'IS': IS,}
