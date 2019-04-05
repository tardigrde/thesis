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


def create_outputs(dir_path, kalmaned):
    result_lists, matrices = extract_attributes_from_saver_objects(kalmaned)

    shp_dir, fig_dir, file_count = check_folders(dir_path)

    export_results_to_shp(result_lists, shp_dir, file_count)

    export_outputs_to_plots(fig_dir, result_lists, matrices, file_count)


def export_outputs_to_plots(fig_dir, result_lists, matrices, file_count):
    start_time = time.time()

    if not file_count: return

    # for i in ['cv', 'ca']:
    #     fig_dir_path = Path(str(fig_dir) + '\\' + file_count)
    #     if not fig_dir_path.is_dir(): makedirs(fig_dir_path)
    #     create_plots(result_lists, matrices, fig_dir_path, file_count,i)

    # plotter.plot_adapted_result(fig_dir, 'adapted', result_lists)

    print("Plotting took %s seconds " % (time.time() - start_time))


def create_plots(result, matrices, fig_dir_path, file_count, type):
    plotter.plot_result(fig_dir_path, type, result)

    # plotter.plot_P(fig_dir_path, matrices)
    #
    # plotter.plot_K(fig_dir_path, matrices)

    plotter.plot_llh(fig_dir_path, type, result)

    plotter.plot_epsilons(fig_dir_path, type, result)

    # plotter.plot_m(fig_dir_path, measurements_count, e, n, d, lat, lng)
    # plotter.plot_P2(fig_dir_path, P, end_count)
    # # plotter.plot_x(fig_dir_path, end_count)
    # plotter.plot_xy(fig_dir_path)
    # # plotter.plot_ned_acc()


def export_results_to_shp(result, shp_dir, file_count):
    og_coords_path = Path(str(shp_dir) + r'\og_coordinates.shp')

    if not og_coords_path.is_file():
        df = pd.DataFrame([[i, j] for i, j in zip(result['oglng'], result['oglat'])])
        df.columns = ['lng', 'lat']
        write_to_shp(df, og_coords_path)

    for i in ['cv', 'ca', 'adapted']:
        df = get_dataframes(result[i])
        shape_file_name = Path(str(shp_dir) + form_filename(i, file_count))
        write_to_shp(df, shape_file_name)


def get_dataframes(type_of_result):
    df = [[i, j] for i, j in zip(type_of_result['lng'], type_of_result['lat'])]
    df = pd.DataFrame(df)
    df.columns = ['lng', 'lat']
    return df


def form_filename(type, file_count):
    result_type = str(type) + '_result'
    filename = str(r'\\' + result_type + str(file_count) + r'.shp')
    return filename


def extract_attributes_from_saver_objects(kalmaned):
    adapted_states = [i for i in kalmaned['adapted_states']]
    saver_cv = kalmaned['saver_cv']._DL
    saver_ca = kalmaned['saver_ca']._DL

    result = {
        'cv': {
            'lng': [st[0] for st in saver_cv['x_post']],
            'lat': [st[2] for st in saver_cv['x_post']],
            'likelihood': saver_cv['likelihood'],
            'epsilons': [e for e in saver_cv['epsilon']],
            'priolng': [ln[0] for ln in kalmaned['saver_cv']['x_prior']],
            'priolat': [lt[0] for lt in kalmaned['saver_cv']['x_prior']],
        },
        'ca': {
            'lng': [st[0] for st in saver_ca['x_post']],
            'lat': [st[2] for st in saver_ca['x_post']],
            'likelihood': saver_ca['likelihood'],
            'epsilons': [e for e in saver_ca['epsilon']],
            'priolng': [ln[0] for ln in kalmaned['saver_cv']['x_prior']],
            'priolat': [lt[0] for lt in kalmaned['saver_cv']['x_prior']],
        },
        'oglng': [st[0] for st in saver_cv['z']],
        'oglat': [st[1] for st in saver_cv['z']],
        'adapted': {
            'lng': [i[1][0] for i in kalmaned['adapted_states']],
            'lat': [i[1][1] for i in kalmaned['adapted_states']],
        }
    }
    matrices = {
        # 'likelihood': saved.likelihood,
        # 'K': saved.K,
        # 'P': saved.P_post,
        # 'length': len(saved._dt),
    }

    return result, matrices


def check_folders(dir_path):
    output_dir_path = Path(str(dir_path) + r'\results')
    shapes_dir = Path(str(output_dir_path) + r'\shapes')
    figures_dir = Path(str(output_dir_path) + r'\figures')

    if not output_dir_path.is_dir(): makedirs(output_dir_path)
    if not shapes_dir.is_dir(): makedirs(shapes_dir)
    if not figures_dir.is_dir(): makedirs(figures_dir)

    file_count = get_file_count(shapes_dir)

    return shapes_dir, figures_dir, file_count


def get_file_count(dir_path):
    onlyfiles = [f for f in listdir(str(dir_path)) if isfile(join(str(dir_path), f))]
    result_count = []
    for file in onlyfiles:
        if 'result' in file and '.shp' in file and 'result.shp' not in file:
            # we want to use the *number* after the result filename substring
            number = re.findall(r'\d+', file)
            result_count.append(int(number[0]))

    if not result_count:
        count = str(0)
    else:
        count = str(max(sorted(result_count)) + 1)
    return count


def write_to_shp(df, out_path):
    start_time = time.time()
    df['Coordinates'] = list(zip(df.lng, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
    print(out_path)
    gdf.to_file(driver='ESRI Shapefile', filename=out_path)
    print("Writing shapes took %s seconds " % (time.time() - start_time))

# def write_result_to_json(json_dir, data):
#     with open(json_dir, 'w') as outfile:
#         json.dump(data, outfile)
