# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION. IT IS A BLACK BOX NOW.

"""

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

# base_dir = r'teszt/'
# current_test_set_dir = r'szeged_trolli_teszt/'
# path_imu = '*.csv'
# path_gps = '*.log'
# acc, gyro, magnetometer, timestamp inputfile

path_imu = r'D:/PyCharmProjects/thesis/teszt/szeged_trolli_teszt/mlt_20181210_163817_190.csv'
path_gps = r'D:/PyCharmProjects/thesis/teszt/szeged_trolli_teszt/nmea.log'
metadata = {
    'created': '20181210_163817',
    'vehicle': "trolleybus",
    'intent': 'developing'
}

measurement = Measurement(path_imu, path_gps, metadata)
measurement.preprocess()
measurement.do_kalman_filtering()
measurement.segment_data()



print("--- %s seconds ---" % (time.time() - start_time))
