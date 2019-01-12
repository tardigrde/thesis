"""
CC-BY-SA2.0 Lizenz
"""
import matplotlib.pyplot as plt
from kalman import Kalman
import nmea_parser
import imu_data_parser
import numpy as np
import initital_parameters
import pandas as pd
from shapely.geometry import Point
import pandas as pd
import geopandas
from shapely.geometry import Point


def get_acceleration_data(path_imu):
    imu_list_of_dicts = imu_data_parser.get_imu_dictionary(path_imu)
    return imu_list_of_dicts


def get_gps_data(path_gps):
    gps_list_of_dicts = nmea_parser.get_gps_dictionary(path_gps)
    return gps_list_of_dicts


def pass_gps_list(gps):
    time, ln, la, v, t, hdop = [], [], [], [], [], []
    timestamps = sorted(list(gps.keys()))
    for t in timestamps:
        values = gps[t]
        v.append(values['v'])
        hdop.append(values['hdop'])
        time.append(values['time'])
        ln.append(values['lng'])
        la.append(values['lat'])
    return {'ln': ln, 'la': la, 'v': v, 't': t, 'hdop': hdop, 'time': time, }


def pass_acc_list(acc):
    timestamps = sorted(list(acc.keys()))
    acc_time, acc_east, acc_north, acc_down = [], [], [], []
    for t in timestamps:
        values = acc[t]
        acc_east.append(values['acc_east'])
        acc_north.append(values['acc_north'])
        acc_down.append(values['acc_down'])
        acc_time.append(values['time'])
    return {'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down, 'acc_time': acc_time}


def _pass_std_devs(acc):
    acc_lists = pass_acc_list(acc)
    std_dev_acc_east = np.std(acc_lists['acc_east'])
    std_dev_acc_north = np.std(acc_lists['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


def interpolate_and_trim_gps_data(acc, gps):
    gps_lists = pass_gps_list(gps)
    acc_lists = pass_acc_list(acc)

    acc_time = acc_lists['acc_time']
    acc_east = acc_lists['acc_east']
    acc_north = acc_lists['acc_north']
    acc_down = acc_lists['acc_down']

    gps_time = gps_lists['time']
    gps_lng = gps_lists['ln']
    gps_lat = gps_lists['la']
    gps_hdop = gps_lists['hdop']
    gps_v = gps_lists['v']

    interpolated_lng = np.interp(acc_time, gps_time, gps_lng).tolist()
    interpolated_lat = np.interp(acc_time, gps_time, gps_lat).tolist()
    interpolated_hdop = np.interp(acc_time, gps_time, gps_hdop).tolist()
    interpolated_v = np.interp(acc_time, gps_time, gps_v).tolist()
    # interpolated_vlng = np.interp(acc_time, gps_time, gps_vln).tolist()
    # interpolated_vlat = np.interp(acc_time, gps_time, gps_vla).tolist()
    # interpolated_t = np.interp(acc_time, gps_time, gps_t).toList()

    """
    TODO:
        - refactor this messssssssssss
    """

    if (len(acc_time) == len(acc_down) == len(acc_east) == len(acc_north) == len(interpolated_lat) == len(
            interpolated_lng) == len(interpolated_v) == len(interpolated_hdop)):
        count = len(acc_time)
        print('LEN DOES EQUALS;\nit is: ', count)
        trim_count = int(round(count / 20, 0))
        print('LEN/20 IS: ', trim_count)

        del (acc_time[:trim_count])
        del (acc_east[:trim_count])
        del (acc_north[:trim_count])
        del (acc_down[:trim_count])
        del (interpolated_lat[:trim_count])
        del (interpolated_lng[:trim_count])
        del (interpolated_hdop[:trim_count])
        del (interpolated_v[:trim_count])

        print('TRIMMED COUNT: ', len(acc_east))

        return {
            'acc_time': acc_time, 'lng': interpolated_lng, 'lat': interpolated_lat,
            'v': interpolated_v, 'hdop': interpolated_hdop,
            'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down
        }
    else:
        print('NEM')


def _predict(X_minus, P_minus, A, Q):
    """
    :param X_minus:
    :param P_minus:
    :param A:
    :param Q:
    :return:
    """
    # Project the state ahead
    x = A * X_minus

    # Project the error covariance ahead
    P = A * P_minus * A.T + Q

    return x, P


def _update(X_predicted, P_predicted, R, I, H, lng, lat, a_east, a_north, sigma_pos, sigma_acc):
    R = np.matrix([[sigma_pos, 0.0, 0.0, 0.0],
                   [0.0, sigma_pos, 0.0, 0.0],
                   [0.0, 0.0, sigma_acc, 0.0],
                   [0.0, 0.0, 0.0, sigma_acc]])

    # Compute the Kalman Gain
    S = H * P_predicted * H.T + R
    K = (P_predicted * H.T) * np.linalg.pinv(S)

    # Update the estimate via z
    Z = np.array([lng, lat, a_east, a_north]).reshape(H.shape[0], 1)
    y = Z - (H * X_predicted)  # Innovation or Residual
    x = X_predicted + (K * y)

    # Update the error covariance
    P = (I - (K * H)) * P_predicted

    return x, P


def get_kalmaned_coordinates(path_acc, path_gps):
    # http: // geopandas.org /

    acc = get_acceleration_data(path_acc)
    gps = get_gps_data(path_gps)

    std_devs = _pass_std_devs(acc)

    init_params = initital_parameters.get_initial_params()
    X = init_params['X']
    P = init_params['P']
    H = init_params['H']
    R = init_params['R']
    A = init_params['F']
    I = init_params['I']
    Q = init_params['Q']
    # Multiplying Q with std_devs
    std_dev = (std_devs['std_dev_acc_east'] + std_devs['std_dev_acc_east']) / 2
    Q = Q * std_dev ** 2

    interpolated_attribute_table = interpolate_and_trim_gps_data(acc, gps)
    acc_time = interpolated_attribute_table['acc_time']
    acc_east = interpolated_attribute_table['acc_east']
    acc_north = interpolated_attribute_table['acc_north']
    acc_down = interpolated_attribute_table['acc_down']
    gps_lng = interpolated_attribute_table['lng']
    gps_lat = interpolated_attribute_table['lat']
    gps_v = interpolated_attribute_table['v']
    gps_hdop = interpolated_attribute_table['hdop']

    print('v0s', (gps_v.count(0) + gps_v.count(1)))

    # Measurements
    og_coordinates = {
        'lng': gps_lng,
        'lat': gps_lat,
    }
    result_coordinates = {
        'lng': [],
        'lat': [],
    }
    measurements_count = len(acc_time)
    # allocation
    x_minus = []
    P_minus = []
    x_list = []
    is_first_step = 1

    for a_east, a_north, lat, lng, v, hdop in zip(acc_east, acc_north, gps_lat, gps_lng, gps_v, gps_hdop):
        # If velocity is lower than 1.5 m/s, we skipp the step
        """
        TODO:
        - idea1: if for more than lets say 500(5 secs) measurments are near 0,
                we start another kalman filter from the next value
        - idea2: leave the 0 velocity measurments as they are, but then do something with the
        """
        if (v <= 1):
            continue
        sigma_pos = hdop
        sigma_acc = std_dev
        if is_first_step:
            is_first_step = 0
            x_minus = np.matrix([[lng, lat, 0.0, 0.0, a_east, a_north]]).T
            P_minus = P

        # Time Update (Prediction)
        # ========================
        x_predicted, P_predicted = _predict(x_minus, P_minus, A, Q)

        # Measurement Update (Correction) AND IT BECOMES LAST STATE
        x_minus, P_minus = _update(x_predicted, P_predicted, R, I, H, lng, lat, a_east, a_north, sigma_pos, sigma_acc)
        x_list.append(x_minus)
        result_coordinates['lng'].append(float(x_minus[0]))
        result_coordinates['lat'].append(float(x_minus[1]))

        # Save states for Plotting
        # savestates(x, Z, P, K)
    print('END', len(result_coordinates['lng']))

    og_df = pd.DataFrame(og_coordinates)
    df = pd.DataFrame(result_coordinates)

    # plot_result(og_coordinates, result_coordinates)
    res_shp_path = r"teszt/szeged_trolli_teszt/shapes/"
    output_file_path = form_filename_dynamically(res_shp_path)

    convert_result_to_shp(og_df, r"teszt/szeged_trolli_teszt/shapes/og_coordinates.shp")
    convert_result_to_shp(df, output_file_path)

    # plot_m(measurements_count, acc_east, acc_north, acc_down, gps_lng, gps_lat)
    # plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'ro')
    # plt.show()
    return


def form_filename_dynamically(dir_path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    result_count = []
    for file in onlyfiles:
        if 'result' in file and '.shp' in file and 'result.shp' not in file:
            # we want to use the number after the result filename substring
            number = file.split('.')[0][-1:]
            result_count.append(int(number))

    count = str(max(sorted(result_count)) + 1)
    filepath = dir_path + 'result' + count + '.shp'
    return filepath


def convert_result_to_shp(df, out_path):
    df['Coordinates'] = list(zip(df.lng, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
    gdf.to_file(driver='ESRI Shapefile', filename=out_path)


def plot_result(og, res):
    fig_gps = plt.figure(figsize=(16, 16))
    plt.scatter(res['lng'], res['lat'])
    plt.scatter(og['lng'], og['lat'])
    plt.xlabel(r'LNG $g$')
    plt.ylabel(r'LAT $g$')
    plt.grid()
    """
    TODO:
    -   make this dynamic so new files are created on every run
    """
    plt.savefig('teszt\szeged_trolli_teszt\Kalman-Filter-RESULTS.png', dpi=72, transparent=True, bbox_inches='tight')


def plot_m(measurements_count, ma_e, ma_n, acc_down, mp_lng, mp_lat):
    fig_acc = plt.figure(figsize=(16, 9))
    plt.step(range(measurements_count), ma_e, label='$a_x$')
    plt.step(range(measurements_count), ma_n, label='$a_y$')
    plt.step(range(measurements_count), acc_down, label='$a_z$')
    plt.ylabel(r'Acceleration $g$')
    plt.ylim([-2, 2])
    plt.legend(loc='best', prop={'size': 18})

    plt.savefig('Kalman-Filter-CA-Acceleration-Measurements.png', dpi=72, transparent=True, bbox_inches='tight')

    fig_gps = plt.figure(figsize=(16, 16))
    # plt.rcParams['figure.facecolor'] = 'white'
    # fig_gps.patch.set_facecolor('white')
    plt.scatter(mp_lng, mp_lat)
    plt.xlabel(r'LNG $g$')
    plt.ylabel(r'LAT $g$')
    plt.grid()
    plt.savefig('Kalman-Filter-CA-GPS-Measurements.png', dpi=72, transparent=True, bbox_inches='tight')


# plot_P()
#
# plot_P2()
#
# plot_K()
#
# plot_x()
#
# plot_xy()

# dist = np.cumsum(np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2))
# print('Your drifted %dm from origin.' % dist[-1])

# Preallocation for Plotting
# xt = []
# yt = []
# dxt = []
# dyt = []
# ddxt = []
# ddyt = []
# Zx = []
# Zy = []
# Px = []
# Py = []
# Pdx = []
# Pdy = []
# Pddx = []
# Pddy = []
# Kx = []
# Ky = []
# Kdx = []
# Kdy = []
# Kddx = []
# Kddy = []
#
# def savestates(x, Z, P, K):
#     xt.append(float(x[0]))
#     yt.append(float(x[1]))
#     dxt.append(float(x[2]))
#     dyt.append(float(x[3]))
#     ddxt.append(float(x[4]))
#     ddyt.append(float(x[5]))
#     Zx.append(float(Z[0]))
#     Zy.append(float(Z[1]))
#     Px.append(float(P[0, 0]))
#     Py.append(float(P[1, 1]))
#     Pdx.append(float(P[2, 2]))
#     Pdy.append(float(P[3, 3]))
#     Pddx.append(float(P[4, 4]))
#     Pddy.append(float(P[5, 5]))
#     Kx.append(float(K[0, 0]))
#     Ky.append(float(K[1, 0]))
#     Kdx.append(float(K[2, 0]))
#     Kdy.append(float(K[3, 0]))
#     Kddx.append(float(K[4, 0]))
#     Kddy.append(float(K[5, 0]))

def do_pothole_extraction(acc, gps):
    """

    :param acc: Dict of lists of acceleration data.
    :param gps: Dict of lists of gps data.
    :return:
    """
    interpolated_attribute_table = interpolate_and_trim_gps_data(acc, gps)
    acc_time = interpolated_attribute_table['acc_time']
    acc_east = interpolated_attribute_table['acc_east']
    acc_north = interpolated_attribute_table['acc_north']
    acc_down = interpolated_attribute_table['acc_down']
    gps_lng = interpolated_attribute_table['lng']
    gps_lat = interpolated_attribute_table['lat']
    gps_v = interpolated_attribute_table['v']
    gps_hdop = interpolated_attribute_table['hdop']
    print('Count of acc_down is {}'.format(len(acc_down)))
    print('Biggest value is {}'.format(max(acc_down)))
    print('Lowest value is {}'.format(min(acc_down)))

    out_weka = './teszt/szeged_trolli_teszt/nointerpolation/for_weka.csv'
    with open(out_weka, 'w') as out:
        for d in acc_down:
            out.write(str(d) + ',' + '' + '\n')
    return

    p_lng = []
    p_lat = []
    # for lng, lat, down in zip(gps_lng, gps_lat, acc_down):
    #     if down > 1.2 or 0.4 < down < 0.8:
    #         p_lng.append(lng)
    #         p_lat.append(lat)
    # pothole = [value for value in acc_down if value > 1.2 or 0.4 < value < 0.8]
    # print('Count of pothole is {}'.format(len(pothole)))
    # plt.plot(acc_down, 'b--')
    # plt.show()
    fig = plt.figure()
    plt.plot(gps_lng, gps_lat, 'b--', p_lng, p_lat, 'rs')
    plt.show()
    fig.savefig('first_potholes_manually_extracted.pdf', dpi=fig.dpi)
    return acc_down


"""
--------------------------------------------------------------------------------------------
"""
"""
    TODO:
    - check/adjust velocity
    - try 2x1D
    - make it OO for real
    - make cross-terms 0
    - try like this: https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1
    - try it like it was before interpolation kinda ok, but not really
    - geohash 


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
