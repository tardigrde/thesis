from lib.madgwick.madgwickahrs import MadgwickAHRS
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from dsp_library import dsp
from utils import fuser, plotter

import pandas as pd
import numpy as np
import math


def apply_low_pass(df):
    dataset = {}
    for col in list(df):
        if col != 'time' and not 'mag' in col:
            low_passed = dsp.do_low_pass_filter(df[col].tolist())
            dataset[col] =  low_passed #df[col].tolist()
            # We already checked visually if the new measurements are better with LPF
            # if col == 'acc_z':
            #     plotter.valid_plot_acc_axis(df[col].tolist(),low_passed)
        else:
            assert col in ['time', 'mag_x', 'mag_y', 'mag_z']
            dataset[col] = df[col].tolist()

    return dataset


def _wrangle_data_with_pandas(path):
    """

    Args:
        path:

    Returns:

    """
    # Transform the huge csv to dataframe.
    df = pd.DataFrame(pd.read_csv(path, sep=','))
    columns_to_keep = ['Timestamp',
                       'accelX', 'accelY', 'accelZ',
                       'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)',
                       'calMagX', 'calMagY', 'calMagZ']
    trimmed_df = df[columns_to_keep]
    new_colum_names = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    pd.options.mode.chained_assignment = None
    trimmed_df.rename(columns=dict(zip(columns_to_keep, new_colum_names)), inplace=True)
    trimmed_df.update(trimmed_df['time'].apply(_milisecondify))
    dataframe = trimmed_df.where(trimmed_df != 0, 0.001)

    return dataframe


def _milisecondify(t):
    """
    We just turn the timestamps from human-readable format to timestamps, representing


    Args:
        t:

    Returns:

    """
    t = (str(t).split(' '))[1].replace(':', '.').split('.')
    ms = int(t[3]) + (int(t[2]) * 1000) + (int(t[1]) * 60 * 1000) + (int(t[0]) * 60 * 60 * 1000)
    return ms


def _calculate_true_acceleration(acceleration):
    """

    Args:
        acceleration:

    Returns:

    """
    acc_east = acceleration[0]
    acc_north = acceleration[1]
    acc_down = acceleration[2] - 1
    magnetic_declination_offset = 5.05000000  # CSONGRÁD MEGYE

    sin_magnetic_declination = math.sin(math.radians(magnetic_declination_offset))
    cos_magnetic_declination = math.cos(math.radians(magnetic_declination_offset))

    eastern_north_component = sin_magnetic_declination * acc_east  # row[0] is acceleration east
    northern_east_component = -sin_magnetic_declination * acc_north  # row[1] is acceleration north

    northern_north_component = cos_magnetic_declination * acc_north
    eastern_east_component = cos_magnetic_declination * acc_east

    acc_e = eastern_east_component + eastern_north_component
    acc_n = northern_north_component + northern_east_component

    return {'east': acc_e, 'north': acc_n, 'down': acc_down}


def _iterate_through_table_and_do_calculations(data):
    """

    Args:
        df:

    Returns:

    """

    # plt.plot(data['acc_z'])
    # plt.show()

    # This will be used! In the next years ;)
    from ImuMeasurement import ImuMeasurement
    im = ImuMeasurement(data)


    time = data['time']
    no_of_measurements = len(time)

    list_of_dicts_of_imu_data = []
    imu_data_dict = {}

    # rolls, pitches, yaws = [], [], []
    # plt.plot(data['acc_x'], color="red")
    # plt.plot(data['acc_y'], color="blue")
    # plt.plot(data['acc_z'], color="green")
    # plt.show()

    # Iterating through rows.
    for i in iter(range(no_of_measurements)):
        acc = [data['acc_x'][i], data['acc_y'][i], data['acc_z'][i]]
        gyro = [data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i]]
        mag = [data['mag_x'][i], data['mag_y'][i], data['mag_z'][i]]

        # Transform acceleration, gyroscope and magnetometer readings to a quaternion.
        MadgwickAHRS.update(MadgwickAHRS, gyro, acc, mag)
        Q = MadgwickAHRS.quaternion

        quaternion = Quaternion([float(Q[0]), float(Q[1]), float(Q[2]), float(Q[3])])
        # quaternion = [float(Q[0]), float(Q[1]), float(Q[2]), float(Q[3])]

        # Transform quaternion to rotation matrix.
        rotation_matrix = quaternion.rotation_matrix.tolist()

        # Experimenting with two different types of roll pitch and yaw
        # They are not consistent
        # roll, pitch, yaw = Q.to_euler_angles()
        # rolls.append(roll)
        # pitches.append(pitch)
        # yaws.append(yaw)
        # _yaw, _pitch, _roll = quaternion.yaw_pitch_roll
        # # INCONSISTENCY
        # print(roll, pitch, yaw)
        # print(_roll, _pitch, _yaw)

        # we have a problem here because the link below states:
        # "So multiplication R*[0 0 1].T() should give us vector A."
        # link: https://blog.maddevs.io/reduce-gps-data-error-on-android-with-kalman-filter-and-accelerometer-43594faed19c
        # def check_rotmat_validity(rotmat, acc):
        #     rotmat_grav = np.dot(rotmat, [0,0,1])
        #     print(list(rotmat_grav),acc)
        #     try:
        #         assert list(rotmat_grav)==acc
        #     except:
        #         pass

        # check_rotmat_validity(rotation_matrix, acc)

        # Calculate absolute acceleration in terms of East-North-Down.
        east_north_down_acceleration = np.dot(
            np.linalg.inv(rotation_matrix),
            [float(data['acc_x'][i]), float(data['acc_y'][i]), float(data['acc_z'][i])]
        )

        # check_correctness_of_transformation

        # print('Ned_acc {}'.format(ned_acc))

        # Rotate absolute acceleration in respect to true north.
        tru_acc = _calculate_true_acceleration(east_north_down_acceleration)

        # print('True_acc {}'.format(tru_acc))

        imu_data_dict[int(time[i])] = {'time': time[i], 'acc_east': tru_acc['east'], 'acc_north': tru_acc['north'],
                                       'acc_down': tru_acc['down']}


        list_of_dicts_of_imu_data.append(
            {'time': int(time[i]), 'acc_east': tru_acc['east'], 'acc_north': tru_acc['north'],
             'acc_down': tru_acc['down']}
        )
    return imu_data_dict


def get_imu_dataframe(imu_data_dict):
    """

    Args:
        imu_data_dict:

    Returns:

    """
    imu_dataframe = pd.DataFrame(imu_data_dict)
    return imu_dataframe


def pass_acc_dict_of_lists(acc):
    timestamps = sorted(list(acc.keys()))
    acc_time, acc_east, acc_north, acc_down = [], [], [], []
    for t in timestamps:
        values = acc[t]
        acc_east.append(values['acc_east'])
        acc_north.append(values['acc_north'])
        acc_down.append(values['acc_down'])
        acc_time.append(int(values['time']))
    # plt.figure(1)
    # plt.plot(acc_east, color="red", label="east-west")
    # plt.plot(acc_north, color="blue",label="north-south")
    # plt.plot(acc_down, color="green",label="down-up")
    # plt.legend()
    # plt.show()

    return {'east': acc_east, 'north': acc_north, 'down': acc_down, '_time': acc_time}


# Call this and on the result of this you can call get_gps_dataframe.
def get_imu_dictionary(path, data='lists'):
    """

    Args:
        path:

    Returns:

    """
    dataframe = _wrangle_data_with_pandas(path)
    imu_data_dict = _iterate_through_table_and_do_calculations(dataframe)

    if data == 'lists':
        return pass_acc_dict_of_lists(imu_data_dict)
    elif data == 'df':
        return get_imu_dataframe(imu_data_dict)
    elif data == 'dicts':
        return imu_data_dict

    # def avg(axis):
    #     return np.average(axis)
    # print(avg(acc_x), avg(data['acc_y']), avg(data['acc_z']), avg(gyro_x), avg(gyro_y), avg(gyro_z), avg(mag_x), avg(mag_y), avg(mag_z))
