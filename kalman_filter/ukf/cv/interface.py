from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from utils.auxiliary import syncronize_timestamps
import numpy as np

def do_ukf(acc, gps):
    print(acc.keys())
    print(gps.keys())

    print(acc['_time'])
    print(gps['time'])

    at, east, north = acc['_time'], acc['east'], acc['north']
    gt, ln, la, hdop = gps['time'], gps['ln'], gps['la'], gps['hdop']
    dt = 0.01
    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
    ukf = UKF(dim_x=4, dim_z=2, fx=f_cv,
              hx=h_cv, dt=dt, points=sigmas)
    ukf.R = np.diag([0.09, 0.09])

    pass