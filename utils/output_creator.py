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
import os


def create_outputs(dir_path, res_obj, type):
    testset = ''
    if os.path.isdir(dir_path):
        testset = os.path.basename(dir_path)
    else:
        print('NO OUTPUTS CREATED')
        return

    shapes_dir, figures_dir, kalmaned_dir, potholes_dir, file_count = check_folders(dir_path)

    if type == 'kalmaned':
        result_lists, matrices = extract_attributes_from_saver_objects(res_obj)
        create_kf_shapes(result_lists, kalmaned_dir, testset, file_count)
        create_kalmaned_plots(figures_dir, result_lists, matrices, file_count)
    elif type == 'potholes':
        file_count = str(int(file_count)-1)
        create_potholes_shapes(res_obj, potholes_dir, testset, file_count)
        export_potholes_plots(res_obj, figures_dir, file_count)


def create_potholes_shapes(res_obj, potholes_dir, testset, file_count):
    type = 'potholes'
    filename = form_filename(type, testset, file_count)
    shape_file_name = Path(str(potholes_dir) + filename)
    gdf = get_geo_dataframe(res_obj)
    write_to_shp(gdf, shape_file_name)


def export_potholes_plots(res_obj, figures_dir, file_count):
    pass


def create_kf_shapes(result_lists, kalmaned_dir, testset, file_count):
    og_coords_path = Path(str(kalmaned_dir) + r'\og_coordinates.shp')
    if not og_coords_path.is_file():
        create_og_outputs(og_coords_path, result_lists)

    # for i in ['cv', 'ca', 'adapted']:
    for type in ['adapted']:
        filename = form_filename(type, testset, file_count)
        shape_file_name = Path(str(kalmaned_dir) + filename)
        gdf = get_geo_dataframe(result_lists[type])
        write_to_shp(gdf, shape_file_name)


def create_kalmaned_plots(fig_dir, result_lists, matrices, file_count):
    # start_time = time.time()

    if not file_count: return
    plotter.plot_adapted_result(fig_dir, 'adapted', result_lists)

    epsilons = result_lists['adapted']['epsilons']
    plotter.plot_epsilons(fig_dir, 'adapted', epsilons)
    # for i in ['cv', 'ca']:
    #     fig_dir_path = Path(str(fig_dir) + '\\' + file_count)
    #     if not fig_dir_path.is_dir(): makedirs(fig_dir_path)
    #     create_plots(result_lists, matrices, fig_dir_path, file_count,i)
    # print("Plotting took %s seconds " % (time.time() - start_time))


def create_og_outputs(og_coords_path, result):
    df = pd.DataFrame([[i, j] for i, j in zip(result['oglng'], result['oglat'])])
    df.columns = ['lng', 'lat']
    df['Coordinates'] = list(zip(df.lng, df.lat))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
    write_to_shp(gdf, og_coords_path)


#####################################################################
def form_filename(type, testset, file_count):
    filename = str(r'\\' + str(testset) + '_' + str(type) + str(file_count) + r'.shp')
    return filename


def get_geo_dataframe(res_obj):
    validity = 'llh'
    datatable = [[i, j, k, l] for i, j, k, l in
                 zip(res_obj['lng'], res_obj['lat'], res_obj['time'], res_obj[validity])]
    df = pd.DataFrame(datatable)
    df.columns = ['lng', 'lat', 'time', validity]

    df['coordinates'] = list(zip(df.lng, df.lat))
    df['coordinates'] = df['coordinates'].apply(Point)

    crs = {'init': 'epsg:23700'}
    gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='coordinates')

    return gdf


def write_to_shp(gdf, out_path):
    print(out_path)
    gdf.to_file(driver='ESRI Shapefile', filename=out_path)


def get_file_count(dir_path):
    onlyfiles = [f for f in listdir(str(dir_path)) if isfile(join(str(dir_path), f))]
    result_count = []
    for file in onlyfiles:
        if not 'og_coordinates' in file:
            # we want to use the *number* after the result filename substring
            number = re.findall(r'\d+', file)
            result_count.append(int(number[0]))

    if not result_count:
        count = str(0)
    else:
        count = str(max(sorted(result_count)) + 1)
    return count


def check_folders(dir_path):
    output_dir_path = Path(str(dir_path) + r'\results')
    shapes_dir = Path(str(output_dir_path) + r'\shapes')
    kalmaned_dir = Path(str(shapes_dir) + r'\kalmaned')
    potholes_dir = Path(str(shapes_dir) + r'\potholes')
    figures_dir = Path(str(output_dir_path) + r'\figures')

    if not output_dir_path.is_dir(): makedirs(output_dir_path)
    if not shapes_dir.is_dir(): makedirs(shapes_dir)
    if not kalmaned_dir.is_dir(): makedirs(kalmaned_dir)
    if not potholes_dir.is_dir(): makedirs(potholes_dir)
    if not figures_dir.is_dir(): makedirs(figures_dir)

    file_count = get_file_count(kalmaned_dir)

    return shapes_dir, figures_dir, kalmaned_dir, potholes_dir, file_count


def extract_attributes_from_saver_objects(kalmaned):
    adapted_states = [i for i in kalmaned['adapted']]
    saver_cv_ukf = kalmaned['ucv']._DL
    saver_cv_kf = kalmaned['cv']._DL
    saver_ca_ukf = kalmaned['uca']._DL
    time = [i['time'] for i in kalmaned['adapted']]

    result = {
        'cv': {
            'lng': [st[0] for st in saver_cv_kf['x_post']],
            'lat': [st[2] for st in saver_cv_kf['x_post']],
            'likelihood': saver_cv_kf['likelihood'],
            'epsilons': [e for e in saver_cv_kf['epsilon']],
            'priolng': [ln[0] for ln in kalmaned['cv']['x_prior']],
            'priolat': [lt[2] for lt in kalmaned['cv']['x_prior']],
        },

        'ucv': {
            'lng': [st[0] for st in saver_cv_ukf['x_post']],
            'lat': [st[2] for st in saver_cv_ukf['x_post']],
            'likelihood': saver_cv_ukf['likelihood'],
            'epsilons': [e for e in saver_cv_ukf['epsilon']],
            'priolng': [ln[0] for ln in kalmaned['ucv']['x_prior']],
            'priolat': [lt[2] for lt in kalmaned['ucv']['x_prior']],
        },

        'uca': {
            'lng': [st[0] for st in saver_ca_ukf['x_post']],
            'lat': [st[2] for st in saver_ca_ukf['x_post']],
            'likelihood': saver_ca_ukf['likelihood'],
            'epsilons': [e for e in saver_ca_ukf['epsilon']],
            'priolng': [ln[0] for ln in kalmaned['uca']['x_prior']],
            'priolat': [lt[2] for lt in kalmaned['uca']['x_prior']],
        },
        'oglng': [st[0] for st in saver_cv_ukf['z']],
        'oglat': [st[1] for st in saver_cv_ukf['z']],
        'adapted': {
            'time': time,
            'lng': [i['x'][0] for i in kalmaned['adapted']],
            'lat': [i['x'][2] for i in kalmaned['adapted']],
            'P': [i['P']for i in kalmaned['adapted']],
            'llh': [i['llh'] for i in kalmaned['adapted']],
            'epsilons': [i['eps'] for i in kalmaned['adapted']],
            'type': [i['model'] for i in kalmaned['adapted']],
        },
        'smoothed':{
            'lng': kalmaned['smoothed'][:,0],
            'lat': kalmaned['smoothed'][:, 2]
        }
    }
    matrices = {
        # 'likelihood': saved.likelihood,
        # 'K': saved.K,
        # 'P': saved.P_post,
        # 'length': len(saved._dt),
    }

    return result, matrices


def create_plots(result, matrices, fig_dir_path, file_count, type):
    # plotter.plot_result(fig_dir_path, type, result)
    # plotter.plot_llh(fig_dir_path, type, result)
    # plotter.plot_epsilons(fig_dir_path, type, result)

    # plotter.plot_P(fig_dir_path, matrices)
    # plotter.plot_K(fig_dir_path, matrices)
    # plotter.plot_m(fig_dir_path, measurements_count, e, n, d, lat, lng)
    # plotter.plot_P2(fig_dir_path, P, end_count)
    # # plotter.plot_x(fig_dir_path, end_count)
    # plotter.plot_xy(fig_dir_path)
    # # plotter.plot_ned_acc()
    pass

# def write_potholes_to_shp(dir_path, potholes):
#
#     df = pd.DataFrame(potholes)
#     df['Coordinates'] = list(zip(df.lng, df.lat))
#     df['Coordinates'] = df['Coordinates'].apply(Point)
#     crs = {'init': 'epsg:23700'}
#     gdf = geopandas.GeoDataFrame(df, crs=crs, geometry='Coordinates')
#     write_to_shp(gdf,out_path)
