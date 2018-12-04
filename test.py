# -*- coding: utf-8 -*-

from interface import Interface
from point import Point
import time

start_time = time.time()

# acc, gyro, magnetometer, timestamp inputfile
#in_imu = '/home/acer/Desktop/szakdoga/code/code_oop/teszt/mlt-20180718-212142-452.csv'
in_imu = './teszt/roszke-szeged/mlt_20180915_150741_18m-1.csv'
# gps nmea sentences inputfile
in_gps = './teszt/roszke-szeged/nmea.log'

# get filtered data
interface = Interface(in_imu, in_gps)

gps = interface.get_gps_data()
acc = interface.get_acceleration_data()

kalmaned_coordinates = interface.do_pykalmaning(acc, gps);

#kalmaned_coordinates = interface.get_kalmaned_coordinates(gps, acc);

# kalmaned_with_ts = interface.pass_kalmaned_list(kalmaned_coordinates, gps)
# acc_with_ts = interface.pass_acc_list(acc)
#print(len(kalmaned_coordinates))
#print(kalmaned_coordinates[0]['IM'])
"""
points = Point()
interpolated_attribute_table = points.interpolate_kalmaned_coordinates(acc_with_ts, kalmaned_with_ts)
print('Len int_att_tabl: %s' % len(interpolated_attribute_table))
"""




#interpolated = points.interpolate_kalmaned_coordinates(acc, gps, kalmaned_coordinates)
#print(interpolated)

print("--- %s seconds ---" % (time.time() - start_time))


