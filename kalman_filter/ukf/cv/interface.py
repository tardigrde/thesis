from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np

def do_ukf(acc, gps):
    # TODO:
    # finish breaking into lists and prepear data for batch filtering til the set is over -->20:00
    print(acc.keys())
    print(gps.keys())

    print(acc['_time'])
    print(gps['time'])
    zs = [[i,j] for i, j in zip(gps['la'][0], gps['ln'][0])]
    dt = 1
    std = 2.
    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)

    ukf = UKF(dim_x=4, dim_z=2, fx=f_cv,
              hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([zs[0][0], 0., zs[0][1], 0.])
    ukf.R = np.diag([std ** 2, std ** 2])
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

    xs, _ = ukf.batch_filter(zs)
    print(xs)


def f_cv(x, dt):
    """ state transition function for a
    constant velocity aircraft"""

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return np.dot(F, x)


def h_cv(x):
    return x[[0, 2]]