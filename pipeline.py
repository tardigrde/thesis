# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION.

"""

from os import listdir
from os.path import isfile, join
from pathlib import Path
from measurement import Measurement
from dsp_library import dsp
import time
import sys
import logging

log = logging.getLogger("logger")

start_time = time.time()
log.info('Setting paths and finding files...')

DATA_BASE_DIR = r'D:\code\PyCharmProjects\thesis\data'
CURRENT_TEST_RUN = r'\trolli_playground'
dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN

# DATA_BASE_DIR = r'D:\code\PyCharmProjects\thesis\data'
# CURRENT_TEST_RUN = r'\20190115\elso'
# dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN


# DATA_BASE_DIR = r'D:\code\PyCharmProjects\thesis\data'
# CURRENT_TEST_RUN = r'\20190409\harmadik'
# dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN

path_to_reference_potholes = r'D:\code\PyCharmProjects\thesis\data\real_potholes\potholes_3m_buffer.shp'

input_dir_path = Path(dir_path + '\measurement')

path_imu = ''
path_gps = ''
stats = {}

date_of_measurement = ''
if input_dir_path.is_dir():
    only_files = [f for f in listdir(input_dir_path) if isfile(join(input_dir_path, f))]
    for file in only_files:
        if '.csv' in file:
            path_imu = Path(str(input_dir_path) + '\\' + file)
            date_of_measurement = file.split('.')[0]
            print('Path of the IMU file: ', path_imu)
        elif '.log' in file:
            path_gps = Path(str(input_dir_path) + '\\' + file)
            print('Path of the GPS file: ', path_gps)

assert not None in (path_imu, path_gps)

log.info('Setting paths and finding files...DONE!')

log.info("Initializing Measurement object...")
max_eps= 10
min_no_of_pothole_like_measurements = 20

measurement = Measurement(path_imu, path_gps, dir_path, path_to_reference_potholes,max_eps, min_no_of_pothole_like_measurements)

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

measurement.evaluate_potholes()
measurement.create_outputs('potholes')

# potholes = dsp.classify_windows(acc, gps, dir_path)
# stats = measurement.get_stats()
# except Exception as e:
#     print('An error occured:', e)

# stats["created"] = date_of_measurement
# stats["vehicle"] = "Ford Fiesta"
# stats["intent"] = "development"
#
#
# path_stats = Path(str(dir_path) + r'\results\metadata.txt')
# with open(path_stats, 'w') as out:
#     out.write(json.dumps(stats, indent=4))

print("--- %s seconds ---" % (time.time() - start_time))

# from utils.auxiliary import fix_shapes
# fix_shapes(DATA_BASE_DIR)
