from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
import math
import time
import sys


def save_adapted_state(filters, epsilons):
    min = math.inf
    lowest = None

    # We search for the lowest epsilon
    for i, eps in enumerate(epsilons):
        if eps is not None and eps < min:
            min = eps
            lowest = i

    state = filters[lowest].x_post
    llh = filters[lowest].likelihood
    eps = epsilons[lowest]
    P = filters[lowest].P

    return {'x': state, 'P': P, 'llh': llh, 'eps': eps, 'model': lowest}


def get_stats(adapted_states):
    types = []
    epsilons = []
    below_three = []
    sum_of_epsilons = .0
    for state in adapted_states:
        epsilon = state['eps']
        # epsilons.append(epsilon)
        type = state['model']
        types.append(type)
        sum_of_epsilons += epsilon
        if epsilon < 3:
            below_three.append(epsilon)
    kf_cv = types.count(0)
    kf_ca = types.count(1)
    ukf_cv = types.count(2)
    ukf_ca = types.count(3)
    print('CVd: {}, CAd: {}, UCVd: {}, UCAd: {}'.format(kf_cv, kf_ca, ukf_cv, ukf_ca))
    print('Length of epsilons below three: ', len(below_three))
    print('The sum of epsilons are:', sum_of_epsilons)


def adjust_process_noise(filters, epsilons, max_eps, Q_scale_factor):
    count = 0

    for e, f in zip(epsilons, filters):
        if not None in (e, f) and e > max_eps:
            f.Q *= Q_scale_factor
            count += 1
        elif not None in (e, f) and count > 0:
            f.Q /= Q_scale_factor
            count -= 1


def do_cv_k_filtering(point, kf_cv, saver_cv_kf):
    z = [point.lng, point.vlng, point.lat, point.vlat]

    kf_cv.R = get_R(point.hdop)

    for i, t in enumerate(point.acc['_time']):
        acc = {
            'north': point.acc['north'][i],
            'east': point.acc['east'][i]
        }
        kf_cv.predict(u=[acc['east'], acc['north']])
    kf_cv.update(z)
    saver_cv_kf.save()

    return kf_cv, saver_cv_kf


def do_ca_k_filtering(point, kf_cv, saver_cv_kf):
    z = [point.lng, point.vlng, point.lat, point.vlat]

    kf_cv.R = get_R(point.hdop)

    # for i, t in enumerate(point.acc['_time']):
    #     acc = {
    #         'north': point.acc['north'][i],
    #         'east': point.acc['east'][i]
    #     }
    #     kf_cv.predict(u=[acc['east'], acc['north']])
    kf_cv.update(z)
    saver_cv_kf.save()

    return kf_cv, saver_cv_kf


def do_cv_uk_filtering(point, ukf_cv, saver_cv_ukf):
    std = point.hdop
    ukf_cv.R = np.diag([std ** 2, std ** 2])

    z = [point.lng, point.lat]

    ukf_cv.predict()
    ukf_cv.update(z)
    saver_cv_ukf.save()
    return ukf_cv, saver_cv_ukf


def do_ca_uk_filtering(point, ukf_ca, saver_ca_ukf):
    z = [point.lng, point.lat]
    std = point.hdop
    ukf_ca.R = np.diag([std** 2, std** 2])
    updated = False
    for i, acc_t in enumerate(point.acc['_time']):
        kwargs = {
            'north': point.acc['north'][i],
            'east': point.acc['east'][i]
        }
        ukf_ca.predict(fx=f_ca, **kwargs)

        if acc_t == point.time:
            ukf_ca.update(z)
            updated = True

    # this is actually the ultimate timestamp check
    assert updated == True
    # print(point.time)
    # print('ERROR: no update happened!')

    saver_ca_ukf.save()

    return ukf_ca, saver_ca_ukf


def set_cv_filter(kf_cv):
    dt = 0.01
    kf_cv.F = get_F(dt, dim=4)
    kf_cv.H = get_H(model='cv')

    kf_cv.B = get_B(dt)
    kf_cv.Q = Q_discrete_white_noise(4, dt=dt, var=0.2)
    # kf_cv.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.02)
    # kf_cv.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.02)

    kf_cv.P = get_P(100)

    return kf_cv


def set_ca_filter(kf_ca):
    dt = 0.01
    kf_ca.F = get_F(dt, dim=6)
    kf_ca.H = get_H(model='ca', process='predict')

    kf_ca.Q = get_Q(kf_ca.Q, dt, var=0.2, dim=6)

    kf_ca.P = get_P(100)

    return kf_ca


def set_ukf_cv_filter(ukf_cv):
    dt = 1
    ukf_cv.Q = get_Q(ukf_cv.Q, dt, var=0.2, dim=4)

    return ukf_cv


def set_ukf_ca_filter(ukf_ca):
    dt = 0.01
    ukf_ca.Q = get_Q(ukf_ca.Q, dt, var=0.1, dim=4)
    # print('Q:', ukf_ca.Q)
    return ukf_ca


######################################################################################
def get_Q(Q, dt, var=0.02, dim=None, ):
    if dim == 4:
        Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=var)
        Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=var)
    elif dim == 6:
        Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=var)
        Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=var)
    else:
        return
    return Q


def get_F(dt, dim=4):
    if dim == 4:
        F = np.array([[1., dt, 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., dt],
                      [0., 0., 0., 1.]])
    elif dim == 6:
        F = np.array([[1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2, 0.0],
                      [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    else:
        return
    return F


def get_P(num):
    P = np.eye(4) * num
    return P


def get_B(dt):
    B = np.array([[(dt ** 2) / 2, 0.],
                  [dt, 0.],
                  [(dt ** 2) / 2, 0.],
                  [0., dt]])
    return B


def get_H(model='cv', process=None):
    if model == 'cv':
        H = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        # H = np.array([[1., 0., 0., 0.],
        #               [0., 0., 1., 0.]])
    elif model == 'ca':
        if process == 'predict':
            H = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        elif process == 'update':
            H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        else:
            return
    else:
        return
    return H


def get_R(stddevs):
    R = np.eye(4) * (stddevs ** 2)
    return R


def f_ca(x, dt, **kwargs):
    """ state transition function for a
    constant velocity aircraft"""
    north = None
    east = None
    if kwargs is not None:
        for key, value in kwargs.items():
            # print ("%s == %s" % (key, value))
            if key == 'north':
                north = value
            else:
                east = value

    # B = np.zeros((4, 2))
    # B[0, 0], B[2, 1] = (dt ** 2) / 2, (dt ** 2) / 2
    # B[1, 1], B[3, 1] = dt, dt
    # B[0, 0], B[2, 0] = (dt ** 2) / 2, (dt ** 2) / 2
    # B[1, 0], B[3, 1] = dt, dt

    B = np.array([[(dt ** 2) / 2, 0],
                  [0, dt],
                  [(dt ** 2) / 2, 0],
                  [0, dt]])

    # LNG-->EAST; LAT--> NORTH
    u = np.transpose([east, north])

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])

    return np.dot(F, x) + np.dot(B, u)


def h_ca(x):
    return x[[0, 2]]
    # return x


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
