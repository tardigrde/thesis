# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION.
WARNING FITERPY--> KF: I changed epsilon
"""

from os import listdir
from os.path import isfile, join
from pathlib import Path
from measurement import Measurement
from dsp_library import dsp
import time
import sys
import logging
from Evaluator import read_shape_file
from geojson import Point, Feature, FeatureCollection, dump

log = logging.getLogger("logger")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

start_time = time.time()
log.info('Setting paths and finding files...')

# path_to_reference_potholes = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes_3m_buffer.shp'
path_to_reference_potholes = r'D:\code\PyCharmProjects\thesis\data\real_potholes\road_anomalies_3m_buffer.shp'
# path_to_reference_potholes = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes_3m_buffer.shp'
path_to_invalid_potholes = r'D:\code\PyCharmProjects\thesis\data\real_potholes\invalid_areas.shp'
path_to_szentharomsag=r'D:\code\PyCharmProjects\thesis\data\real_potholes\szentharomsag_uttest.shp'
precisions = []


def write_to_shape(hit_counts):

    # from shapely.geometry import mapping, Polygon
    schema = {
        'geometry': 'Polygon',
        'properties': {'count': 'int'},
    }

    output = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes_hit_3m.geojson'
    feature_collection = FeatureCollection(hit_counts)
    with open(output,'w') as out:
        dump(feature_collection, out)



def get_measurement_path(test_run):
    DATA_BASE_DIR = r'D:\code\PyCharmProjects\thesis\data'
    CURRENT_TEST_RUN = test_run
    dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN
    return dir_path


def get_true_positives_on_pothole(precisions):
    print('NAH', len(precisions), ' ', precisions)
    if len(precisions) == 1: return
    prc = precisions
    overalls = []
    shapes = read_shape_file(path_to_reference_potholes)
    potholes_count_by_reference = []
    for i, j, k, l, shp in zip(prc[0], prc[1], prc[2], prc[3], shapes):
        overall = i + j + k + l
        overalls.append(overall)
        hit_count = {
            'geometry': shp['geometry'],
            'properties': {'count': overall},
        }

        potholes_count_by_reference.append(hit_count)
    write_to_shape(potholes_count_by_reference)
    print(overalls)
    print(sorted(overalls))


def get_paths(dir_path):
    path_imu = None
    path_gps = None
    input_dir_path = Path(dir_path + '\measurement')
    if not input_dir_path.is_dir(): return

    only_files = [f for f in listdir(input_dir_path) if isfile(join(input_dir_path, f))]
    for file in only_files:
        if '.csv' in file:
            path_imu = Path(str(input_dir_path) + '\\' + file)
            print('Path of the IMU file: ', path_imu)
        elif '.log' in file:
            path_gps = Path(str(input_dir_path) + '\\' + file)
            print('Path of the GPS file: ', path_gps)

    return path_imu, path_gps


def run_algorithm(path_imu, path_gps, dir_path, max_eps, min_no_of_pothole_like_measurements):
    log.info('Setting paths and finding files...DONE!')

    log.info("Initializing Measurement object...")


    measurement = Measurement(path_imu, path_gps, dir_path, path_to_reference_potholes, path_to_invalid_potholes,path_to_szentharomsag,
                              max_eps, min_no_of_pothole_like_measurements)

    log.info("Initializing Measurement object...DONE!")
    log.info("Preprocessing...")

    measurement.preprocess()

    log.info("Preprocessing...DONE!")
    log.info("Initializing Point objects...")

    measurement.get_points()

    log.info("Initializing Point objects...DONE!")
    log.info("Kalman filtering...")

    measurement.do_kalman_filtering()

    measurement.create_outputs('kalmaned')
    log.info("Kalman filtering...DONE!")

    prec = measurement.evaluate_potholes()
    precisions.append(prec)
    measurement.create_outputs('potholes')


# for test_run in [r'\20190115\elso']:
# for test_run in [r'\roszke-szeged']:
# for test_run in [r'\20190115\masodik', r'\20190409\elso', r'\20190409\masodik', r'\20190409\harmadik']:
for test_run in [r'\trolli_playground']:
    print('Processing: ', test_run)
    dir_path = get_measurement_path(test_run)
    path_imu, path_gps = get_paths(dir_path)
    assert not None in (path_imu, path_gps)
    max_eps = 5
    min_no_events_in_window = 10
    config={
        #TODO:
        #   input data and parameters unified
    }
    run_algorithm(path_imu, path_gps, dir_path, max_eps, min_no_events_in_window)

get_true_positives_on_pothole(precisions)

print("--- %s seconds ---" % (time.time() - start_time))
