from shapely.geometry import Point
from os import listdir, makedirs
from os.path import isfile, join
from pathlib import Path
from utils import plotter
import numpy as np
import pandas as pd
import geopandas
import time




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
    std_dev_acc_east = np.std(acc['acc_east'])
    std_dev_acc_north = np.std(acc['acc_north'])
    return {'std_dev_acc_east': std_dev_acc_east, 'std_dev_acc_north': std_dev_acc_north}


def interpolate_and_trim_data(acc_lists, gps_lists):
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
    five_percent = int(round(count / 20, 0))

    for c in columns:
        # Checking if all the lists has the same size
        if not len(dataset[c]) == count: return
        count = len(dataset[c])
        del (dataset[c][:five_percent], dataset[c][-five_percent:])

    return dataset


def create_outputs(dir_path, og_coordinates, result, end_count, P_minus, measurements_count, e, n, d, lat, lng):
    og_df = pd.DataFrame(og_coordinates)
    df = pd.DataFrame(result)
    shp_dir, fig_dir = check_folders(dir_path)
    og_coords_path = Path(str(shp_dir) + r'\og_coordinates.shp')

    if not og_coords_path.is_file(): convert_result_to_shp(og_df, og_coords_path)

    file_count = form_filename_dynamically(shp_dir)
    shape_file_path = Path(str(shp_dir) + r'\result' + file_count + r'.shp')
    convert_result_to_shp(df, shape_file_path)

    do_plotting(fig_dir, file_count, og_coordinates, result, end_count, P_minus, measurements_count, e, n, d, lat, lng)


def check_folders(dir_path):
    output_dir_path = Path(str(dir_path) + r'\results')
    shapes_dir = Path(str(output_dir_path) + r'\shapes')
    figures_dir = Path(str(output_dir_path) + r'\figures')

    if not output_dir_path.is_dir(): makedirs(output_dir_path)
    if not shapes_dir.is_dir(): makedirs(shapes_dir)
    if not figures_dir.is_dir(): makedirs(figures_dir)
    return shapes_dir, figures_dir


def form_filename_dynamically(dir_path):
    onlyfiles = [f for f in listdir(str(dir_path)) if isfile(join(str(dir_path), f))]
    result_count = []
    for file in onlyfiles:
        if 'result' in file and '.shp' in file and 'result.shp' not in file:
            # we want to use the number after the result filename substring
            number = int(file.split('.')[0].split('t')[1])
            result_count.append(number)
        # elif not 'og_coordinates' in file:

    if not result_count:
        count = str(0)
    else:
        count = str(max(sorted(result_count)) + 1)
    return count


def convert_result_to_shp(df, out_path):
    start_time = time.time()
    df['Coordinates'] = list(zip(df.lng, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
    gdf.to_file(driver='ESRI Shapefile', filename=out_path)
    print("Writing shapes took %s seconds " % (time.time() - start_time))


def do_plotting(fig_dir, file_count, og_coordinates, result, end_count, P, measurements_count, e, n, d, lat, lng):
    start_time = time.time()
    if not file_count: return
    fig_dir_path = Path(str(fig_dir) + '\\' + file_count)

    if not fig_dir_path.is_dir(): makedirs(fig_dir_path)

    plotter.plot_result(fig_dir_path, og_coordinates, result)

    plotter.plot_m(fig_dir_path, measurements_count, e, n, d, lat, lng)

    plotter.plot_P(fig_dir_path, end_count)

    plotter.plot_P2(fig_dir_path, P, end_count)

    plotter.plot_K(fig_dir_path, end_count)

    # plotter.plot_x(fig_dir_path, end_count)

    plotter.plot_xy(fig_dir_path)

    plotter.plot_ned_acc()

    print("Plotting took %s seconds " % (time.time() - start_time))



# def pass_gps_dict_of_lists(gps):
#     time, ln, la, v, t, hdop = [], [], [], [], [], []
#     timestamps = sorted(list(gps.keys()))
#     for t in timestamps:
#         values = gps[t]
#         v.append(values['v'])
#         hdop.append(values['hdop'])
#         time.append(values['time'])
#         ln.append(values['lng'])
#         la.append(values['lat'])
#     return {'ln': ln, 'la': la, 'v': v, 't': t, 'hdop': hdop, 'time': time, }
#
# def pass_acc_dict_of_lists(path_imu):
#     acc = imu_data_parser.get_imu_dictionary(path_imu)
#     timestamps = sorted(list(acc.keys()))
#     acc_time, acc_east, acc_north, acc_down = [], [], [], []
#     for t in timestamps:
#         values = acc[t]
#         acc_east.append(values['acc_east'])
#         acc_north.append(values['acc_north'])
#         acc_down.append(values['acc_down'])
#         acc_time.append(values['time'])
#     return {'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down, 'acc_time': acc_time}
#
# def get_acceleration_data(path_imu):
#     imu_list_of_dicts = imu_data_parser.get_imu_dictionary(path_imu)
#     return imu_list_of_dicts
#
#
# def get_gps_data(path_gps):
#     gps_list_of_dicts = nmea_parser.get_gps_dictionary(path_gps)
#     return gps_list_of_dicts
