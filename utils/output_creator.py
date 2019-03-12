from shapely.geometry import Point
from os import listdir, makedirs
from os.path import isfile, join
from pathlib import Path
from utils import plotter
import pandas as pd
import json
import geopandas
import time
import re


def create_outputs(dir_path, saver):
    # if not len(og_coordinates) == len(saved):
    #     print('Error: there is a mismatch between the length of the inputs and outputs! No outputs will be generated.')
    #     return

    result , matrices= create_needed_attributes_from_saver_object(saver)
    """
    paraaaaaaaa a result objecttel
    """
    df = pd.DataFrame(result)
    print(df.head())
    shp_dir, fig_dir, json_dir = check_folders(dir_path)
    og_coords_path = Path(str(shp_dir) + r'\og_coordinates.shp')

    if not og_coords_path.is_file(): convert_result_to_shp(df, og_coords_path, og=True)

    file_count = form_filename_dynamically(shp_dir)
    shape_file_path = Path(str(shp_dir) + r'\result' + file_count + r'.shp')
    convert_result_to_shp(df, shape_file_path)

    #write_result_to_json(Path(str(shp_dir) + r'\result' + file_count + r'.json'), saver._DL)

    do_plotting(fig_dir, result, matrices, file_count)


def write_result_to_json(json_dir, data):
    with open(json_dir, 'w') as outfile:
        json.dump(data, outfile)


def create_needed_attributes_from_saver_object(saved):
    if saved._dim_z[0] == 2:
        lng = saved.x_post[:, 0]
        lat = saved.x_post[:, 2]
        priolng = saved.x_prior[:, 0]
        priolat = saved.x_prior[:, 2]
        # oglng= saved.z[:, 0]
        # oglat=saved.z[:, 1]
    elif saved._dim_z[0] == 4:
        lng = saved.x_post[:, 0]
        lat = saved.x_post[:, 3]

    result = {
        'lng': lng,
        'lat': lat,
        'priolng': priolng,
        'priolat': priolat,
        'oglng': saved.z[:, 0],
        'oglat': saved.z[:, 1],
        # 'likelihood': saved.likelihood,
        # 'K': saved.K,
        # 'P': saved.P_post,
        # 'length': len(saved._dt),
    }
    matrices= {
        'likelihood': saved.likelihood,
        'K': saved.K,
        'P': saved.P_post,
        'length': len(saved._dt),
    }

    return result, matrices


def check_folders(dir_path):
    output_dir_path = Path(str(dir_path) + r'\results')
    shapes_dir = Path(str(output_dir_path) + r'\shapes')
    figures_dir = Path(str(output_dir_path) + r'\figures')
    jsons_dir = Path(str(output_dir_path) + r'\jsons')

    if not output_dir_path.is_dir(): makedirs(output_dir_path)
    if not shapes_dir.is_dir(): makedirs(shapes_dir)
    if not figures_dir.is_dir(): makedirs(figures_dir)
    if not jsons_dir.is_dir(): makedirs(jsons_dir)
    return shapes_dir, figures_dir, jsons_dir


def form_filename_dynamically(dir_path):
    onlyfiles = [f for f in listdir(str(dir_path)) if isfile(join(str(dir_path), f))]
    result_count = []
    for file in onlyfiles:
        if 'result' in file and '.shp' in file and 'result.shp' not in file:
            # we want to use the number after the result filename substring
            # number = int(file.split('.')[0].split('t')[1])
            number = re.findall(r'\d+', file)
            result_count.append(int(number[0]))

    if not result_count:
        count = str(0)
    else:
        count = str(max(sorted(result_count)) + 1)
    return count


def convert_result_to_shp(df, out_path, og=False):
    start_time = time.time()
    if og == True:
        df['Coordinates'] = list(zip(df.oglng, df.oglat))
    else:
        df['Coordinates'] = list(zip(df.lng, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
    print(out_path)
    gdf.to_file(driver='ESRI Shapefile', filename=out_path)
    print("Writing shapes took %s seconds " % (time.time() - start_time))


def do_plotting(fig_dir, result, matrices, file_count):
    start_time = time.time()
    if not file_count: return
    fig_dir_path = Path(str(fig_dir) + '\\' + file_count)

    if not fig_dir_path.is_dir(): makedirs(fig_dir_path)

    plotter.plot_result(fig_dir_path, result)

    plotter.plot_P(fig_dir_path, matrices)

    # plotter.plot_m(fig_dir_path, measurements_count, e, n, d, lat, lng)
    #
    # plotter.plot_P2(fig_dir_path, P, end_count)
    #
    print(matrices['K'][0])
    plotter.plot_K(fig_dir_path, matrices)
    #
    # # plotter.plot_x(fig_dir_path, end_count)
    #
    # plotter.plot_xy(fig_dir_path)
    #
    # # plotter.plot_ned_acc()

    print("Plotting took %s seconds " % (time.time() - start_time))
