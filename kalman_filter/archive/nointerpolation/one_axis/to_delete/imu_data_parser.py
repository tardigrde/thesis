from lib.madgwick.madgwickahrs import MadgwickAHRS
from pyquaternion import Quaternion
from utils import fuser
import pandas as pd
import numpy as np
import math


def _wrangle_data_with_pandas(path):
    """

    Args:
        path:

    Returns:

    """
    # Transform the huge csv to dataframe.
    df = pd.DataFrame(pd.read_csv(path, sep=','))
    trimmed_df = df[['Timestamp',
                     'accelX', 'accelY', 'accelZ',
                     'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)',
                     'calMagX', 'calMagY', 'calMagZ']]
    pd.options.mode.chained_assignment = None
    trimmed_df.update(trimmed_df['Timestamp'].apply(_milisecondify))
    dataframe = trimmed_df.where(trimmed_df != 0, 0.001)

    return dataframe


def _milisecondify(t):
    """

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
    magnetic_declination_offset = 4.96666667  # CSONGRÁD MEGYE

    sin_magnetic_declination = math.sin(math.radians(magnetic_declination_offset))
    cos_magnetic_declination = math.cos(math.radians(magnetic_declination_offset))

    eastern_north_component = sin_magnetic_declination * acc_east  # row[0] is acceleration east
    northern_east_component = -sin_magnetic_declination * acc_north  # row[1] is acceleration north

    northern_north_component = cos_magnetic_declination * acc_north
    eastern_east_component = cos_magnetic_declination * acc_east

    acc_e = eastern_east_component + eastern_north_component
    acc_n = northern_north_component + northern_east_component

    return {'east': acc_e, 'north': acc_n, 'down': acc_down}


def _iterate_through_table_and_do_calculations(df):
    """

    Args:
        df:

    Returns:

    """
    # Extracting columns.
    time = [str(t) for t in df.iloc[:, 0]]
    acc_x = [aX for aX in df.iloc[:, 1]]
    acc_y = [aY for aY in df.iloc[:, 2]]
    acc_z = [aZ for aZ in df.iloc[:, 3]]
    gyro_x = [gX for gX in df.iloc[:, 4]]
    gyro_y = [gY for gY in df.iloc[:, 5]]
    gyro_z = [gZ for gZ in df.iloc[:, 6]]
    mag_x = [gX for gX in df.iloc[:, 7]]
    mag_y = [gY for gY in df.iloc[:, 8]]
    mag_z = [gZ for gZ in df.iloc[:, 9]]
    no_of_measurements = len(time)

    list_of_dicts_of_imu_data = []
    imu_data_dict = {}
    # Iterating through rows.
    for i in range(no_of_measurements):
        acc = [acc_x[i], acc_y[i], acc_z[i]]
        gyro = [gyro_x[i], gyro_y[i], gyro_z[i]]
        mag = [mag_x[i], mag_y[i], mag_z[i]]

        # Transform acceleration, gyroscope and magnetometer readings to a quaternion.
        MadgwickAHRS.update(MadgwickAHRS, gyro, acc, mag)
        Q = MadgwickAHRS.quaternion
        quaternion = Quaternion([float(Q[0]), float(Q[1]), float(Q[2]), float(Q[3])])

        # Transform quaternion to rotation matrix.
        rotation_matrix = quaternion.rotation_matrix.tolist()

        # Calculate absolute acceleration in terms of East-North-Down.
        ned_acc = np.dot(np.linalg.inv(rotation_matrix),
                         np.transpose([float(acc_x[i]), float(acc_y[i]), float(acc_z[i])])).tolist()

        # print('Ned_acc {}'.format(ned_acc))

        # Rotate absolute acceleration in respect to true north.
        tru_acc = _calculate_true_acceleration(ned_acc)

        # print('True_acc {}'.format(tru_acc))

        imu_data_dict[int(time[i])] = {'time': time[i], 'acc_east': tru_acc['east'], 'acc_north': tru_acc['north'],
                                       'acc_down': tru_acc['down']}

        list_of_dicts_of_imu_data.append(
            {'time': int(time[i]), 'acc_east': tru_acc['east'], 'acc_north': tru_acc['north'],
             'acc_down': tru_acc['down']}
        )
    # plotter.plot_xyz_acc(acc)
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
    return {'acc_east': acc_east, 'acc_north': acc_north, 'acc_down': acc_down, 'acc_time': acc_time}

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