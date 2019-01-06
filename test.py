# -*- coding: utf-8 -*-

import interface
import time

start_time = time.time()

# acc, gyro, magnetometer, timestamp inputfile
path_imu = r'teszt/szeged_trolli_teszt/mlt_20181210_163817_190.csv'
path_gps = r'teszt/szeged_trolli_teszt/nmea.log'



gps = interface.get_gps_data(path_gps)
print('Length of output gps data is: {}'.format(len(gps)))
acc = interface.get_acceleration_data(path_imu)
print('Length of acceleration data is: {}'.format(len(acc)))

pothole_count = interface.do_pothole_extraction(acc, gps)

# kalmaned_coordinates = interface.get_kalmaned_coordinates(acc, gps)

print("--- %s seconds ---" % (time.time() - start_time))


#interpolated = points.interpolate_kalmaned_coordinates(acc, gps, kalmaned_coordinates)
#print(interpolated)


#in_imu = '/home/acer/Desktop/szakdoga/code/code_oop/teszt/mlt-20180718-212142-452.csv'
