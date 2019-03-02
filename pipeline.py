# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION. IT IS A BLACK BOX NOW.

"""

from os import listdir
from os.path import isfile, join
from pathlib import Path
from measurement import Measurement
from dsp_library import dsp
import time

start_time = time.time()

"""
TODO:
- make the code into a black box package => 
        class is a measurement which has multipiple properties
        like path_gps and path_imu and date 
- call the package from outside
"""

DATA_BASE_DIR = r'D:\PyCharmProjects\thesis\data'

# CURRENT_TEST_RUN = r'\roszke-szeged'
# CURRENT_TEST_RUN = r'\szeged_trolli_teszt'
CURRENT_TEST_RUN = r'\trolli_playground'

CURRENT_TEST_SET = r'\constacc'

dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN + CURRENT_TEST_SET

input_dir_path = Path(dir_path + '\measurement')

path_imu = ''
path_gps = ''

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



stats = {}

measurement = Measurement(path_imu, path_gps, dir_path)
measurement.preprocess()
#measurement.get_filtered_acc()
#acc, gps = measurement.get_state_of(['acc', 'gps'])
acc, gps = measurement.get_lists_of_measurements();

segmented = dsp.segment_data(acc, gps, dir_path)
# measurement.do_kalman_filtering()
# measurement.segment_data()
# stats = measurement.get_stats()
# except Exception as e:
#     print('An error occured:', e)

stats["created"] = date_of_measurement
stats["vehicle"] = "Ford Fiesta"
stats["intent"] = "development"


path_stats = Path(str(dir_path) + r'\results\metadata.txt')
# with open(path_stats, 'w') as out:
#     out.write(json.dumps(stats, indent=4))

print("--- %s seconds ---" % (time.time() - start_time))
