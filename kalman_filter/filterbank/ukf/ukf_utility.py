from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
import time
import sys


def do_unscented_kalman_filtering(type, points, dir_path):
    return manage_adaptive_filtering(type, points)


def manage_adaptive_filtering(type, points):
    start_time = time.time()

    kalam_filter_results = {}
    adapted_states = []

    # set_of_points_to_filter = points[0]
    ukf_cv, saver_cv, ukf_ca, saver_ca = create_filterbank()

    for i, set_of_points_to_filter in enumerate(points):
        print('Kalman Filter count: {}'.format(i))

        # Setting the first state.
        ukf_cv.x = np.array([points[i][0].lng, 0., points[i][0].lat, 0.])
        ukf_ca.x = np.array([points[i][0].lng, 0., points[i][0].lat, 0.])

        ukf_cv = set_cv_filter(ukf_cv)
        ukf_ca = set_ca_filter(ukf_ca)

        for j, p in enumerate(set_of_points_to_filter):
            if type == 'adaptive':
                ukf_cv, saver_cv, ukf_ca, saver_ca, adapted_state = \
                    do_adaptive_kf(p, ukf_cv, saver_cv, ukf_ca, saver_ca)

                adapted_states.append((p.time, adapted_state))

    saver_cv.to_array()
    saver_ca.to_array()

    kalam_filter_results = {
        'cv_res': ukf_cv, 'saver_cv': saver_cv,
        'ca_res': ukf_ca, 'saver_ca': saver_ca, 'adapted_states': adapted_states
    }
    print("KF took %s seconds " % (time.time() - start_time))
    print('Size of kf result object: ', sys.getsizeof(kalam_filter_results) / 1000, ' MB')

    return kalam_filter_results


def create_filterbank():
    sigmas_cv = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
    ukf_cv = UKF(dim_x=4, dim_z=2, fx=f_cv, hx=h_cv, dt=1, points=sigmas_cv)
    saver_cv = Saver(ukf_cv)

    sigmas_ca = MerweScaledSigmaPoints(4, alpha=.5, beta=2., kappa=1.)
    ukf_ca = UKF(dim_x=4, dim_z=2, fx=f_ca, hx=h_ca, dt=0.01, points=sigmas_ca)
    saver_ca = Saver(ukf_ca)
    return ukf_cv, saver_cv, ukf_ca, saver_ca


def set_cv_filter(ukf_cv, ):
    ukf_cv.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=2)
    ukf_cv.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=2)

    return ukf_cv


def set_ca_filter(ukf_ca, ):
    dt = 0.01

    ukf_ca.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=2)
    ukf_ca.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=2)
    return ukf_ca


def do_adaptive_kf(point, ukf_cv, saver_cv, ukf_ca, saver_ca):
    std = point.hdop ** 2
    ukf_cv.R = np.diag([std ** 2, std ** 2])
    ukf_ca.R = np.diag([std ** 2, std ** 2])

    ukf_cv, saver_cv = do_cv_uk_filtering(point, ukf_cv, saver_cv)
    cv_epsilons = ukf_cv.epsilon

    ukf_ca, saver_ca = do_ca_uk_filtering(point, ukf_ca, saver_ca)
    ca_epsilons = ukf_ca.epsilon

    adapted_state = save_adapted_state(ukf_cv, cv_epsilons, ukf_ca, ca_epsilons)

    return ukf_cv, saver_cv, ukf_ca, saver_ca, adapted_state


def save_adapted_state(ukf_cv, cv_epsilons, ukf_ca, ca_epsilons):
    state_cv = ukf_cv.x_post
    state_ca = ukf_ca.x_post

    if cv_epsilons <= ca_epsilons:
        return [state_cv[0], state_cv[2], cv_epsilons,0]
    elif cv_epsilons > ca_epsilons:
        return [state_ca[0], state_ca[2], ca_epsilons,1]


def do_cv_uk_filtering(point, ukf_cv, saver_cv):
    z = [point.lng, point.lat]

    ukf_cv.predict()
    ukf_cv.update(z)
    saver_cv.save()
    return ukf_cv, saver_cv


def do_ca_uk_filtering(point, ukf_ca, saver_ca):
    z = [point.lng, point.lat]
    for i, t in enumerate(point.acc['_time']):
        kwargs = {
            'north': point.acc['north'][i],
            'east': point.acc['east'][i]
        }
        ukf_ca.predict(fx=f_ca, **kwargs)
    ukf_ca.update(z)
    saver_ca.save()

    return ukf_ca, saver_ca


######################################################################################


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
    print([north, east])

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])

    return (np.dot(F, x) + np.dot(B, u))


def h_ca(x):
    return x[[0, 2]]


def f_cv(x, dt):
    """ state transition function for a
    constant velocity aircraft"""
    # print('______________', x, dt)

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return np.dot(F, x)


def h_cv(x):
    return x[[0, 2]]

# def dummy():
#     for i, etap in enumerate(etaps_acc):
#         ukf.x = np.array([etaps_gps[i][0][1], 0., etaps_gps[i][0][2], 0.])
#         for j, acc_msrmnt in enumerate(etap):
#             kwargs = {'north': acc_msrmnt[1], 'east': acc_msrmnt[2]}
#             ukf.predict(dt, fx=f_ca, **kwargs)
#             try:
#                 if acc_msrmnt[0] == etaps_gps[i][count][0]:
#                     print(j)
#                     print("updated")
#                     measurements = [etaps_gps[i][count][1], etaps_gps[i][count][2]]
#                     ukf.update(measurements)
#                     saver_updated.save()
#                     y, S = ukf.y, ukf.S
#                     eps = np.dot(y.T, inv(S)).dot(y)
#                     epsilons.append(eps)
#                     count = count + 1
#             except Exception as e:
#                 print(e)
#             saver_predicted.save()
#         # epsilons.append(0)
#         count = 0
#
#     print('epsilons', len(epsilons))
#
#     output_creator.create_outputs(dir_path, saver_updated, epsilons)

# def run_filter_bank(point, show_zs=True):
#     dt = 0.1
#     threshold = 3
#     cvfilter = make_cv_filter(dt, std=0.8)
#     cafilter = make_ca_filter(dt, std=0.8)
#     pos, zs = generate_data(120, std=0.8)
#     z_xs = zs[:, 0]
#     xs, res = [], []
#
#     cvfilter.predict()
#     cafilter.predict()
#     cvfilter.update([z])
#     cafilter.update([z])
#
#     std = np.sqrt(cvfilter.R[0, 0])
#     if abs(cvfilter.y[0]) < 2 * std:
#         xs.append(cvfilter.x[0])
#     else:
#         xs.append(cafilter.x[0])
#     res.append(cvfilter.y[0])
#
#     # if show_zs:
#     #     plot_track_and_residuals(dt, xs, z_xs, res)
#     # else:
#     #     plot_track_and_residuals(dt, xs, None, res)
