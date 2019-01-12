# -*- coding: utf-8 -*-

"""
THIS STARTS THE APPLICATION. IT IS A BLACK BOX NOW.

"""

import interface
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
path_imu = r'teszt/szeged_trolli_teszt/mlt_20181210_163817_190.csv'
path_gps = r'teszt/szeged_trolli_teszt/nmea.log'

pothole_count = interface.get_kalmaned_coordinates(path_imu, path_gps)


print("--- %s seconds ---" % (time.time() - start_time))
