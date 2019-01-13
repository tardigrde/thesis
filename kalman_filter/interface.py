"""
CC-BY-SA2.0 Lizenz
"""
import matplotlib.pyplot as plt
from kalman_filter import nmea_parser, initital_parameters, imu_data_parser, kalman
import numpy as np
import pandas as pd
import geopandas
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import peakutils
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


def interpolate_gps_datalists(time, gps_data):
    gps_time = gps_data['time']
    ln = gps_data['ln']
    lt = gps_data['la']
    hdop = gps_data['hdop']
    v = gps_data['v']

    gtime = np.interp(time, gps_time, gps_time).tolist()
    lng = np.interp(time, gps_time, ln).tolist()
    lat = np.interp(time, gps_time, lt).tolist()
    p = np.interp(time, gps_time, hdop).tolist()
    v = np.interp(time, gps_time, v).tolist()

    return gtime, lng, lat, p, v


def _pass_std_devs(acc):
    acc_lists = pass_acc_list(acc)
    std_dev_acc_east = np.std(acc_lists['acc_east'])
    std_dev_acc_north = np.std(acc_lists['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


def find_max_peaks(z_axis, height):
    peaks, _ = find_peaks(z_axis, height)
    return peaks, _


def segment_data(acc, gps):
    dataset = interpolate_and_trim_data(acc, gps)
    z_axis = dataset['down']
    lng = dataset['lng']
    lat = dataset['lat']
    x = np.linspace(0, int(len(z_axis)), len(z_axis) + 1)
    print(x)

    # max, _max = find_peaks(z_axis, 0.05)
    # z_axis_min = [-z for z in z_axis]
    # min, _m = find_peaks(z_axis_min, -0.05)
    # print(len(min))

    max_indexes = peakutils.indexes(z_axis, thres=0.05)
    print(len(max_indexes))
    print(x[max_indexes], z_axis[max_indexes])
    plt.figure(figsize=(10, 6))
    plt.title("peaks")
    plt.plot(x, z_axis)
    # plt.plot(x, z_axis, max_indexes)
    plt.show()

    return max, min


def interpolate_and_trim_data(acc, gps):
    gps_lists = pass_gps_list(gps)
    acc_lists = pass_acc_list(acc)
    c = list(acc_lists.keys())

    time, lng, lat, p, v = interpolate_gps_datalists(acc_lists['acc_time'], gps_lists)

    dataset = {
        'time': acc_lists['acc_time'],
        'east': acc_lists['acc_east'],
        'north': acc_lists['acc_north'],
        'down': acc_lists['acc_down'],
        'gps_time': time,
        'lng': lng,
        'lat': lat,
        'hdop': p,
        'vel': v,
    }
    columns = list(dataset.keys())
    count = len(dataset['time'])
    trim_five_five_percents = int(round(count / 20, 0))

    for c in columns:
        # Checking if all the lists has the same size
        if not len(dataset[c]) == count: return
        count = len(dataset[c])
        del (dataset[c][:trim_five_five_percents], dataset[c][-trim_five_five_percents:])

    return dataset


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


def _update(X_predicted, P_predicted, I, H, lng, lat, a_east, a_north, hdop, std_dev_acc):
    sigma_pos = hdop
    sigma_acc = std_dev_acc

    x, P, Z, K = kalman.kf_update(X_predicted, P_predicted, I, H, lng, lat, a_east, a_north, sigma_pos, sigma_acc)

    return x, P, Z, K


def get_kalmaned_datatable(acc, gps):
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
    std_dev_acc = (std_devs['std_dev_acc_east'] + std_devs['std_dev_acc_east']) / 2
    Q = Q * std_dev_acc

    dataset = interpolate_and_trim_data(acc, gps)
    d = dataset
    # Measurements

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

    for time, a_east, a_north, a_down, lng, lat, v, hdop in zip(d['time'], d['east'], d['north'], d['down'], d['lng'],
                                                                d['lat'], d['vel'],
                                                                d['hdop']):
        # If velocity is lower than 1.5 m/s, we skipp the step
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
        x_minus, P_minus, Z, K = _update(x_predicted, P_predicted, I, H, lng, lat, a_east, a_north, hdop, std_dev_acc)

        x_list.append(x_minus)

        result['lng'].append(float(x_minus[0]))
        result['lat'].append(float(x_minus[1]))
        result['vlg'].append(float(x_minus[2]))
        result['vlt'].append(float(x_minus[3]))
        result['east'].append(float(x_minus[4]))
        result['north'].append(float(x_minus[5]))
        result['down'].append(float(a_down))

        # Save states for Plotting
        # savestates(x_minus, Z, P_minus, K)
    print('END', len(result['lng']))
    print('kalman count', kalman_count)
    end_count = len(result['lng'])

    og_df = pd.DataFrame(og_coordinates)
    df = pd.DataFrame(result)

    plot_result(og_coordinates, result)
    res_shp_path = r"teszt/szeged_trolli_teszt/shapes/"
    output_file_path = form_filename_dynamically(res_shp_path)

    # convert_result_to_shp(og_df, r"teszt/szeged_trolli_teszt/shapes/og_coordinates.shp")
    convert_result_to_shp(df, output_file_path)

    # plot_m(measurements_count, acc_east, acc_north, acc_down, gps_lng, gps_lat)
    # plt.plot(og_lng, og_lat, 'bs', lng_to_plot, lat_to_plot, 'ro')
    # plt.show()
    return result


def form_filename_dynamically(dir_path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    result_count = []
    for file in onlyfiles:
        if 'result' in file and '.shp' in file and 'result.shp' not in file:
            # we want to use the number after the result filename substring
            number = int(file.split('.')[0].split('t')[1])
            result_count.append(number)

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
