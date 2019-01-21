# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION. IT IS A BLACK BOX NOW.

"""

from os import listdir
from os.path import isfile, join
from pathlib import Path
from kalman_filter.measurement import Measurement
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
CURRENT_TEST_RUN = r'\20190115'

CURRENT_TEST_SET = r'\harmadik'

dir_path = DATA_BASE_DIR + CURRENT_TEST_RUN + CURRENT_TEST_SET

input_dir_path = Path(dir_path + r'\measurement')



date_of_measurement = ''
if input_dir_path.isdir():
    only_files = [f for f in listdir(input_dir_path) if isfile(join(input_dir_path, f))]
    for file in only_files:
        if '.csv' in file:
            path_imu = input_dir_path + '\\' + file
            date_of_measurement = file.split('.')[0]
            print('Path of the IMU file: ', path_imu)
        elif '.log' in file:
            path_gps = input_dir_path + '\\' + file
            print('Path of the GPS file: ', path_gps)

# path_imu = r'C:\Users\leven\Desktop\20190115\mlt_20190117_121118_300.csv'

# path_imu = r'D:\PyCharmProjects\thesis\teszt\20190115\harmadik\msrmnt.csv'
# path_gps = r'D:\PyCharmProjects\thesis\teszt\20190115\harmadik\nmea.log'



try:
    measurement = Measurement(path_imu, path_gps, dir_path)
    measurement.preprocess()
    stats = measurement.get_stats()
    measurement.do_kalman_filtering()
    #measurement.segment_data()
except Exception as e:
    print('An error occured:', e)

metadata = {
    'created': date_of_measurement,
    'vehicle': "Ford Fiesta",
    'intent': 'developing',
}



print("--- %s seconds ---" % (time.time() - start_time))
