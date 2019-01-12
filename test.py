# -*- coding: utf-8 -*-

import interface
import time

start_time = time.time()

# acc, gyro, magnetometer, timestamp inputfile
path_imu = r'teszt/szeged_trolli_teszt/mlt_20181210_163817_190.csv'
path_gps = r'teszt/szeged_trolli_teszt/nmea.log'

pothole_count = interface.get_kalmaned_coordinates(path_imu, path_gps)

# kalmaned_coordinates = interface.get_kalmaned_coordinates(acc, gps)

print("--- %s seconds ---" % (time.time() - start_time))

