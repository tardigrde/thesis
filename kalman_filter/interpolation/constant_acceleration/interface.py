"""
CC-BY-SA2.0 Lizenz
"""
from kalman_filter.interpolation.constant_acceleration import kalman, initital_parameters
import numpy as np

from utils import auxiliary
from utils import plotter


def _predict(X_minus, P_minus, A, Q):
    """
    :param X_minus:
    :param P_minus:
    :param A:
    :param Q:
    :return:
    """
    x, P = kalman.kf_predict(X_minus, P_minus, A, Q)

    return x, P


def _update(X_predicted, P_predicted, I, H, lng, lat, a_east, a_north, a_down, hdop, result):
    sigma_pos = hdop
    sigma_acc = 0.053

    x, P, Z, K = kalman.kf_update(X_predicted, P_predicted, I, H, lng, lat, a_east, a_north, sigma_pos, sigma_acc)

    result['lng'].append(float(x[0]))
    result['lat'].append(float(x[1]))
    result['vlg'].append(float(x[2]))
    result['vlt'].append(float(x[3]))
    result['east'].append(float(x[4]))
    result['north'].append(float(x[5]))
    result['down'].append(float(a_down))

    return x, P, Z, K, result


def get_kalmaned_datatable(acc, gps, dir_path):
    init_params = initital_parameters.get_initial_params()
    X = init_params['X']
    P = init_params['P']
    H = init_params['H']
    R = init_params['R']
    A = init_params['F']
    I = init_params['I']
    Q = init_params['Q']

    # Multiplying Q with std_devs

    dataset = auxiliary.interpolate_and_trim_data(acc, gps)
    d = dataset

    og_coordinates = {
        'lng': d['lng'],
        'lat': d['lat'],
    }
    result = {
        'lng': [],
        'lat': [],
        'vlg': [],
        'vlt': [],
        'east': [],
        'north': [],
        'down': [],
    }

    measurements_count = len(d["time"])
    # allocation
    x_minus = []
    P_minus = []
    x_list = []
    unused_count = 0
    measurement_usage = {}
    kalman_count = 0
    is_first_step = 1

    for time, a_east, a_north, a_down, lat, lng, v, hdop in zip(d['time'], d['east'], d['north'], d['down'],
                                                                d['lat'], d['lng'], d['vel'], d['hdop']):
        # If velocity is lower than 2 m/s, we skipp the step
        if (v <= 2):
            if unused_count >= 200:
                measurement_usage[time] = 1
                is_first_step = 1
                unused_count = 0
            else:
                measurement_usage[time] = 0
                unused_count = unused_count + 1
                continue

        """
        TODO:
        - idea1: if for more than lets say 500(5 secs) measurments are near 0,
                we start another kalman filter from the next value
        - idea2: leave the 0 velocity measurments as they are, but then do something with the
        """

        if is_first_step:
            is_first_step = 0
            kalman_count = kalman_count + 1
            x_minus = np.matrix([[lng, lat, 0.0, 0.0, a_east, a_north]]).T
            P_minus = P

        # Time Update (Prediction)
        # ========================
        x_predicted, P_predicted = _predict(x_minus, P_minus, A, Q)

        # Measurement Update (Correction)
        x_minus, P_minus, Z, K, res = _update(x_predicted, P_predicted, I, H, lng, lat, a_east, a_north, a_down, hdop,
                                              result)
        result = res

        x_list.append(x_minus)

        # Save states for Plotting
        plotter.savestates(x_minus, Z, P_minus, K)
    end_count = len(result['lng'])
    print('END', end_count)
    print('kalman count', kalman_count)

    stats = {
        "og_length": measurements_count,
        "used_msrmnts": end_count,
        "og_gps_length": len(gps),
        "kalman_count": kalman_count,
        "unused_count": unused_count,
    }

    auxiliary.create_outputs(dir_path, og_coordinates, result, end_count, P_minus, measurements_count,
                   d['east'], d['north'], d['down'], d['lat'], d['lng'])

    # plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'ro')
    # plt.show()
    return result, stats





# STD_DEV SHOULD ONLY BE COUNTED ON A PHONE IN PEACE!!!
# std_devs = _pass_std_devs(acc)
# std_dev_acc = (std_devs['std_dev_acc_east'] + std_devs['std_dev_acc_east']) / 2
# print('stdv', std_dev_acc)

# dist = np.cumsum(np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2))
# print('Your drifted %dm from origin.' % dist[-1])


# def do_pothole_extraction(acc, gps):
#     """
#
#     :param acc: Dict of lists of acceleration data.
#     :param gps: Dict of lists of gps data.
#     :return:
#     """
#     interpolated_attribute_table = interpolate_and_trim_gps_data(acc, gps)
#     acc_time = interpolated_attribute_table['acc_time']
#     acc_east = interpolated_attribute_table['acc_east']
#     acc_north = interpolated_attribute_table['acc_north']
#     acc_down = interpolated_attribute_table['acc_down']
#     gps_lng = interpolated_attribute_table['lng']
#     gps_lat = interpolated_attribute_table['lat']
#     gps_v = interpolated_attribute_table['v']
#     gps_hdop = interpolated_attribute_table['hdop']
#     print('Count of acc_down is {}'.format(len(acc_down)))
#     print('Biggest value is {}'.format(max(acc_down)))
#     print('Lowest value is {}'.format(min(acc_down)))
#
#     out_weka = './teszt/szeged_trolli_teszt/nointerpolation/for_weka.csv'
#     with open(out_weka, 'w') as out:
#         for d in acc_down:
#             out.write(str(d) + ',' + '' + '\n')
#     return
#
#     p_lng = []
#     p_lat = []
#     # for lng, lat, down in zip(gps_lng, gps_lat, acc_down):
#     #     if down > 1.2 or 0.4 < down < 0.8:
#     #         p_lng.append(lng)
#     #         p_lat.append(lat)
#     # pothole = [value for value in acc_down if value > 1.2 or 0.4 < value < 0.8]
#     # print('Count of pothole is {}'.format(len(pothole)))
#     # plt.plot(acc_down, 'b--')
#     # plt.show()
#     fig = plt.figure()
#     plt.plot(gps_lng, gps_lat, 'b--', p_lng, p_lat, 'rs')
#     plt.show()
#     fig.savefig('first_potholes_manually_extracted.pdf', dpi=fig.dpi)
#     return acc_down


"""
--------------------------------------------------------------------------------------------
"""
"""
    TODO:
    - check/adjust velocity (on nointerpolation branch)
    - try 2x1D
    


    https://github.com/akshaychawla/1D-Kalman-Filter
    https://dsp.stackexchange.com/questions/8860/kalman-filter-for-position-and-velocity-introducing-speed-estimates
    https://dsp.stackexchange.com/questions/38045/even-more-on-kalman-filter-for-position-and-velocity?noredirect=1&lq=1
    https://dsp.stackexchange.com/questions/48343/python-how-can-i-improve-my-1d-kalman-filter-estimate
    https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
    https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
    https://www.reddit.com/r/computervision/comments/35y4kj/looking_for_a_python_example_of_a_simple_2d/
    https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1 !!!!!!!!!!!!!!!!
    https://balzer82.github.io/Kalman/
    https://towardsdatascience.com/kalman-filter-an-algorithm-for-making-sense-from-the-insights-of-various-sensors-fused-together-ddf67597f35e
    https://gist.github.com/manicai/922976
"""
