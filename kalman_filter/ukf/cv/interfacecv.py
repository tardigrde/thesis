from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
import matplotlib.pyplot as plt
from utils.output_creator import create_outputs


def do_ukf(dir_path, acc, gps):
    # TODO:
    # finish breaking into lists and prepear data for batch filtering til the set is over -->20:00
    dt = 1
    std = 2.
    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
    ukf = UKF(dim_x=4, dim_z=2, fx=f_cv,
              hx=h_cv, dt=dt, points=sigmas)
    etaps = []
    for i, j in zip(gps['ln'], gps['la']):
        etaps.append([[k, l] for k, l in zip(i, j)])

    saver = Saver(ukf)

    states = []
    covariances = []
    for i, zs in enumerate(etaps):
        ukf.x = np.array([zs[0][0], 0., zs[0][1], 0.])
        ukf.R = np.diag([std ** 2, std ** 2])
        ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=1)
        ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=1)
        xs, _ = ukf.batch_filter(zs, saver=saver)
        saver.to_array()
        plt.plot(xs[:, 0], xs[:, 2])
        # plt.show()


    create_outputs(dir_path, saver)


def f_cv(x, dt):
    """ state transition function for a
    constant velocity aircraft"""
    print('______________',x,dt)

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return np.dot(F, x)


def h_cv(x):
    return x[[0, 2]]
