from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
import matplotlib.pyplot as plt
from utils.output_creator import create_outputs


def do_ukf_with_acc(dir_path, acc, gps):
    # TODO:
    # finish breaking into lists and prepeare data for batch filtering til the set is over -->20:00
    # data = get_fused_table()
    acct = acc['_time']
    gpst = gps['time']

    dt = 0.01
    std = 2.
    sigmas = MerweScaledSigmaPoints(4, alpha=.5, beta=2., kappa=1.)
    ukf = UKF(dim_x=4, dim_z=2, fx=f_ca,
              hx=h_ca, dt=dt, points=sigmas)
    saver_predicted = Saver(ukf)
    saver_updated = Saver(ukf)

    etaps_gps = []
    for i, j, k in zip(gpst, gps['ln'], gps['la']):
        etaps_gps.append([[l, m, n] for l, m, n in zip(i, j, k)])
    etaps_acc = []
    for at, an, ae in zip(acct, acc['north'], acc['east']):
        etaps_acc.append([[i, j, k] for i, j, k in zip(at, an, ae)])

    count = 0
    states = []
    covariances = []

    ukf.R = np.diag([std ** 2, std ** 2])
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=1)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=1)
    for i, etap in enumerate(etaps_acc):
        ukf.x = np.array([etaps_gps[i][0][1], 0., etaps_gps[i][0][2], 0.])
        for j, acc_msrmnt in enumerate(etap):
            kwargs = {'north': acc_msrmnt[1], 'east': acc_msrmnt[2]}
            ukf.predict(dt, fx=f_ca, **kwargs)
            try:
                if acc_msrmnt[0] == etaps_gps[i][count][0]:
                    print(j)
                    print("updated")
                    measurements=[etaps_gps[i][count][1], etaps_gps[i][count][2]]
                    ukf.update(measurements)
                    saver_updated.save()
                    count = count + 1
            except Exception as e:
                print(e)
            saver_predicted.save()
        count = 0

        # xs, _ = ukf.batch_filter(zs, saver=saver)

        # plt.plot(xs[:, 0], xs[:, 2])
        # plt.show()

    create_outputs(dir_path, saver_updated)


def f_ca(x, dt, **kwargs):
    """ state transition function for a
    constant velocity aircraft"""
    north = 0
    east = 0
    if kwargs is not None:
        for key, value in kwargs.items():
            # print ("%s == %s" % (key, value))
            if key == 'north':
                north = value
            else:
                east = value

    B = np.zeros((4, 2))
    # B[0, 0], B[2, 1] = (dt ** 2) / 2, (dt ** 2) / 2
    # B[1, 1], B[3, 1] = dt, dt
    B[0, 0], B[2, 0] = (dt ** 2) / 2, (dt ** 2) / 2
    B[1, 0], B[3, 1] = dt, dt

    u = np.transpose([north, east])

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])

    return (np.dot(F, x) + np.dot(B, u))


def h_ca(x):
    return x[[0, 2]]
